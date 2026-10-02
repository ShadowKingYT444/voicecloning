export function splitIntoPassages(text, maxWords = 22, minWords = 15) {
  const max = Math.max(15, Math.min(24, Number(maxWords) || 22));
  const minimum = Math.min(minWords, max);
  const paragraphs = String(text)
    .replace(/\r\n?/g, '\n')
    .split(/\n\s*\n/)
    .map((part) => part.trim())
    .filter(Boolean);
  const passages = [];

  for (const paragraph of paragraphs) {
    const words = paragraph.split(/\s+/);
    let cursor = 0;
    const local = [];

    while (cursor < words.length) {
      const hardEnd = Math.min(words.length, cursor + max);
      let end = hardEnd;
      if (hardEnd < words.length) {
        for (let candidate = hardEnd; candidate >= cursor + minimum; candidate -= 1) {
          if (/[.!?;:,]["'”’)]?$/.test(words[candidate - 1])) {
            end = candidate;
            break;
          }
        }
      }
      local.push(words.slice(cursor, end).join(' '));
      cursor = end;
    }

    if (local.length > 1) {
      const tail = local.at(-1);
      const prior = local.at(-2);
      if (wordCount(tail) < minimum && wordCount(prior) + wordCount(tail) <= max) {
        local.splice(-2, 2, `${prior} ${tail}`);
      }
    }
    passages.push(...local);
  }
  return passages;
}

export function wordCount(text) {
  const match = String(text).trim().match(/\S+/g);
  return match ? match.length : 0;
}

export function normalizeWhitespace(text) {
  return String(text).replace(/\s+/g, ' ').trim();
}
