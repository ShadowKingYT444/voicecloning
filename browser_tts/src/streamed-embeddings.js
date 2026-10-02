import { sha256Blob, REVISION, getPinnedAsset } from './model-loader.js';
import { halfToFloat } from './numeric.js';

// The deterministic manifest binds every shard to the exact pinned FP16 tables.
// Rebuilding from those tables must reproduce this digest; no manifest trust-on-first-use.
export const LOSSLESS_MANIFEST_SHA256 = 'a244131d73e89e0b39e06522535eb1267340ff7c27306290f7ec4cca890198da';
const ROW_BYTES = 768 * 2;
const MAX_MANIFEST_BYTES = 256 * 1024;
const DEFAULT_CACHE_BYTES = 4 * 1024 * 1024;

export class StreamedEmbeddings {
  constructor(manifest, manifestUrl, { fetchImpl = globalThis.fetch, cacheBytes = DEFAULT_CACHE_BYTES } = {}) {
    if (!Number.isSafeInteger(cacheBytes) || cacheBytes < 64 * ROW_BYTES || cacheBytes > DEFAULT_CACHE_BYTES) {
      throw Error('The lossless embedding cache must be between one shard and 4 MiB.');
    }
    this.manifest = manifest; this.manifestUrl = manifestUrl;
    // Window/Worker fetch is a Web IDL method; a reader object is not a valid
    // receiver. Bind the original global instead of invoking it as our method.
    this.fetchImpl = fetchImpl.bind(globalThis);
    this.cacheBytes = cacheBytes; this.cache = new Map(); this.residentBytes = 0;
    this.stats = { mode: 'lossless-fp16-shards', cacheBudgetBytes: cacheBytes, cachePeakBytes: 0,
      cacheBytes: 0, fetchedShardBytes: 0, shardRequests: 0, cacheHits: 0,
      sourceTableBytes: 87304704, manifestSha256: LOSSLESS_MANIFEST_SHA256,
      browserMemoryMeasured: false };
  }
  static async load(manifestUrl, { fetchImpl = globalThis.fetch, location = globalThis.location, ...options } = {}) {
    const url = new URL(manifestUrl, location.href);
    if (url.origin !== location.origin || url.username || url.password || !['http:', 'https:'].includes(url.protocol)) {
      throw Error('Lossless embeddings must use the browser app origin.');
    }
    const response = await fetchImpl(url.href);
    if (!response.ok) throw Error(`Could not load the lossless embedding manifest: HTTP ${response.status}.`);
    const blob = await response.blob();
    if (blob.size > MAX_MANIFEST_BYTES || await sha256Blob(blob) !== LOSSLESS_MANIFEST_SHA256) {
      throw Error('Lossless embedding manifest hash/size differs from the pinned export.');
    }
    const manifest = JSON.parse(await blob.text());
    const asset = getPinnedAsset('embed_tokens_fp16');
    if (manifest.schema !== 'voice-study.lossless-embedding-shards/v1' || manifest.revision !== REVISION
      || manifest.source_graph_sha256 !== asset.graphSha256 || manifest.source_data_sha256 !== asset.externalSha256
      || manifest.dtype !== 'float16-le' || manifest.embedding_dim !== 768 || manifest.rows_per_shard !== 64) {
      throw Error('The lossless embedding manifest has an unexpected source contract.');
    }
    return new StreamedEmbeddings(manifest, url.href, { fetchImpl, ...options });
  }
  async shard(table, token) {
    const entry = this.manifest.tables[table].shards[Math.floor(token / 64)];
    const key = entry.file;
    if (this.cache.has(key)) {
      const data = this.cache.get(key); this.cache.delete(key); this.cache.set(key, data);
      this.stats.cacheHits += 1; return { entry, data };
    }
    // Evict before fetch; only one lookup/shard is in flight in the serial worker.
    while (this.residentBytes + entry.bytes > this.cacheBytes) {
      const oldest = this.cache.keys().next().value;
      this.residentBytes -= this.cache.get(oldest).byteLength; this.cache.delete(oldest);
    }
    const response = await this.fetchImpl(new URL(entry.file, this.manifestUrl).href);
    if (!response.ok) throw Error(`Could not load embedding shard ${entry.file}: HTTP ${response.status}.`);
    const blob = await response.blob();
    if (blob.size !== entry.bytes || await sha256Blob(blob) !== entry.sha256) {
      throw Error(`Embedding shard ${entry.file} failed its byte/hash check.`);
    }
    const data = new DataView(await blob.arrayBuffer());
    this.cache.set(key, data); this.residentBytes += data.byteLength;
    this.stats.fetchedShardBytes += blob.size; this.stats.shardRequests += 1;
    this.stats.cacheBytes = this.residentBytes;
    this.stats.cachePeakBytes = Math.max(this.stats.cachePeakBytes, this.residentBytes);
    return { entry, data };
  }
  async lookup(ids) {
    if (!ids.length || ids.length > 2048) throw Error('Embedding input must contain between 1 and 2048 IDs.');
    // Exactly the pinned graph's hybrid Slice(-2)/Where(50256 -> 6561)/Gather/Cast contract.
    const rows = Array.from(ids, (value, index) => {
      let token = Number(value);
      const table = index < ids.length - 2 ? 'text' : 'speech';
      if (table === 'speech' && token === 50256) token = 6561;
      if (!Number.isSafeInteger(token) || token < 0 || token >= this.manifest.tables[table].rows) {
        throw Error(`Invalid ${table} embedding token ${token}.`);
      }
      return { table, token };
    });
    const output = new Float32Array(ids.length * 768);
    for (let row = 0; row < rows.length; row += 1) {
      const { table, token } = rows[row];
      const { entry, data } = await this.shard(table, token);
      const offset = (token - entry.first_row) * ROW_BYTES;
      for (let column = 0; column < 768; column += 1) {
        output[row * 768 + column] = halfToFloat(data.getUint16(offset + column * 2, true));
      }
    }
    return output;
  }
  snapshot() { return { ...this.stats }; }
  clear() { this.cache.clear(); this.residentBytes = 0; this.stats.cacheBytes = 0; }
}
