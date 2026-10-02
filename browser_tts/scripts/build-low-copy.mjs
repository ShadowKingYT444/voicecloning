#!/usr/bin/env node

import { copyFile, link, lstat, mkdir, readdir, rename, rm } from 'node:fs/promises';
import { constants, createReadStream, createWriteStream } from 'node:fs';
import { basename, dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { pipeline } from 'node:stream/promises';
import { randomUUID } from 'node:crypto';
import { build } from 'vite';

const PACKAGE_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const PUBLIC_DIR = join(PACKAGE_ROOT, 'public');
const DIST_DIR = join(PACKAGE_ROOT, 'dist');
const VITE_CONFIG = join(PACKAGE_ROOT, 'vite.config.js');
const STREAM_HIGH_WATER_MARK = 1024 * 1024;
const HARD_LINK_EXTENSIONS = ['.onnx_data', '.onnx.data', '.onnx'];

function isImmutableModelAsset(path) {
  return HARD_LINK_EXTENSIONS.some((extension) => path.endsWith(extension));
}

function temporaryPath(destination) {
  return join(dirname(destination), `.${basename(destination)}.${process.pid}.${randomUUID()}.tmp`);
}

async function copyBounded(source, destination) {
  await pipeline(
    createReadStream(source, { highWaterMark: STREAM_HIGH_WATER_MARK }),
    createWriteStream(destination, { flags: 'wx', highWaterMark: STREAM_HIGH_WATER_MARK }),
  );
}

async function stageFile(source, destination, relativePath, stats) {
  const sourceStat = await lstat(source);
  if (sourceStat.isSymbolicLink() || !sourceStat.isFile()) {
    throw new Error(`Public asset is not a regular file: ${relativePath}`);
  }

  await mkdir(dirname(destination), { recursive: true });
  const temporary = temporaryPath(destination);
  try {
    if (isImmutableModelAsset(relativePath)) {
      try {
        // These hard links share an inode with public. Keep the pinned source assets immutable.
        await link(source, temporary);
        stats.hardLinkedFiles += 1;
        stats.hardLinkedBytes += sourceStat.size;
      } catch (error) {
        if (error?.code !== 'EXDEV') throw error;
        await copyBounded(source, temporary);
        stats.fallbackFiles += 1;
        stats.fallbackBytes += sourceStat.size;
      }
    } else {
      await copyFile(source, temporary, constants.COPYFILE_EXCL);
      stats.copiedFiles += 1;
      stats.copiedBytes += sourceStat.size;
    }
    await rename(temporary, destination);
    return sourceStat.size;
  } catch (error) {
    await rm(temporary, { force: true }).catch(() => {});
    throw error;
  }
}

async function stageDirectory(sourceDirectory, destinationDirectory, relativeDirectory, stats) {
  const sourceStat = await lstat(sourceDirectory);
  if (sourceStat.isSymbolicLink() || !sourceStat.isDirectory()) {
    throw new Error(`Public asset directory is not a regular directory: ${sourceDirectory}`);
  }
  const entries = await readdir(sourceDirectory, { withFileTypes: true });
  entries.sort((left, right) => left.name.localeCompare(right.name));

  for (const entry of entries) {
    const source = join(sourceDirectory, entry.name);
    const destination = join(destinationDirectory, entry.name);
    const relativePath = relativeDirectory ? join(relativeDirectory, entry.name) : entry.name;
    if (entry.isSymbolicLink()) {
      throw new Error(`Public asset symlinks are not allowed: ${relativePath}`);
    }
    if (entry.isDirectory()) {
      await mkdir(destination, { recursive: true });
      await stageDirectory(source, destination, relativePath, stats);
    } else if (entry.isFile()) {
      stats.bytes += await stageFile(source, destination, relativePath, stats);
      stats.files += 1;
    } else {
      throw new Error(`Unsupported public asset entry: ${relativePath}`);
    }
  }
}

async function main() {
  console.log('Running Vite build with public-directory copying disabled.');
  await build({
    configFile: VITE_CONFIG,
    root: PACKAGE_ROOT,
    build: { copyPublicDir: false },
  });

  const stats = {
    files: 0,
    bytes: 0,
    copiedFiles: 0,
    copiedBytes: 0,
    hardLinkedFiles: 0,
    hardLinkedBytes: 0,
    fallbackFiles: 0,
    fallbackBytes: 0,
  };
  await stageDirectory(PUBLIC_DIR, DIST_DIR, '', stats);
  console.log(
    `Staged ${stats.files} public files (${stats.bytes} bytes): `
      + `${stats.hardLinkedFiles} hard links (${stats.hardLinkedBytes} bytes), `
      + `${stats.copiedFiles} copies (${stats.copiedBytes} bytes), `
      + `${stats.fallbackFiles} bounded cross-filesystem copies (${stats.fallbackBytes} bytes).`,
  );
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
