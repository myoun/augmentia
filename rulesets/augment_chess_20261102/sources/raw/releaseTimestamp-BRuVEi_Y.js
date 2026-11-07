const RELEASE_TIMESTAMP_ID = "2026-11-01-independent-draft-expansion-v1";
const RELEASE_TIMESTAMP_DATE = "2026.11.02";
const RELEASE_TIMESTAMP_TIME = "";
const RELEASE_TIMESTAMP_DISPLAY = [RELEASE_TIMESTAMP_DATE, RELEASE_TIMESTAMP_TIME].filter(Boolean).join(" ");
function parseReleaseTimestamp(value) {
  if (typeof value !== "string" || !/^\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}$/.test(value)) return null;
  const [date, time] = value.split(" "), iso = date.replaceAll(".", "-") + "T" + time + ":00+09:00", ms = Date.parse(iso);
  if (!Number.isFinite(ms)) return null;
  const local = new Date(ms + 9 * 60 * 60 * 1e3).toISOString();
  if (local.slice(0, 10) !== date.replaceAll(".", "-") || local.slice(11, 16) !== time) return null;
  return { releaseId: RELEASE_TIMESTAMP_ID, date, time, display: value, dateTime: iso };
}
async function loadPublishedReleaseTimestamp({ fetch: fetcher = globalThis.fetch, timeoutMs = 1500 } = {}) {
  const fallback = () => ({
    releaseId: RELEASE_TIMESTAMP_ID,
    date: RELEASE_TIMESTAMP_DATE,
    time: null,
    display: RELEASE_TIMESTAMP_DISPLAY,
    dateTime: RELEASE_TIMESTAMP_DATE.replaceAll(".", "-")
  });
  if (typeof fetcher !== "function") return fallback();
  const controller = new AbortController();
  let timer;
  try {
    const response = await Promise.race([
      fetcher("/release-info.json", { cache: "no-store", credentials: "same-origin", signal: controller.signal }),
      new Promise((_, reject) => {
        timer = setTimeout(() => {
          controller.abort();
          reject(new Error("Release metadata timeout"));
        }, timeoutMs);
      })
    ]);
    if (!response.ok) return fallback();
    const value = await response.json();
    if (value?.releaseId !== RELEASE_TIMESTAMP_ID) return fallback();
    return parseReleaseTimestamp(`${value.date} ${value.time}`) ?? fallback();
  } catch {
    return fallback();
  } finally {
    clearTimeout(timer);
  }
}
function applyPublishedReleaseTimestamp(root, timestamp) {
  if (!timestamp) return;
  for (const element of root.querySelectorAll("[data-current-release-time]")) {
    element.textContent = timestamp.display;
    element.setAttribute("datetime", timestamp.dateTime);
  }
}
export {
  RELEASE_TIMESTAMP_DATE as R,
  RELEASE_TIMESTAMP_TIME as a,
  applyPublishedReleaseTimestamp as b,
  loadPublishedReleaseTimestamp as l
};
