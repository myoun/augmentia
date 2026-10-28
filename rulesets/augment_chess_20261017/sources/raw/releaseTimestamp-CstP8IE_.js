const RELEASE_TIMESTAMP_ID = "2026-10-17";
function parseReleaseTimestamp(value) {
  if (typeof value !== "string" || !/^2026\.10\.17 (?:[01]\d|2[0-3]):[0-5]0$/.test(value)) return null;
  const [date, time] = value.split(" ");
  return {
    releaseId: RELEASE_TIMESTAMP_ID,
    date,
    time,
    display: value,
    dateTime: `${date.replaceAll(".", "-")}T${time}:00+09:00`
  };
}
async function loadPublishedReleaseTimestamp({ fetch: fetcher = globalThis.fetch, timeoutMs = 1500 } = {}) {
  if (typeof fetcher !== "function") return null;
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
    if (!response.ok) return null;
    const value = await response.json();
    if (value?.releaseId !== RELEASE_TIMESTAMP_ID) return null;
    return parseReleaseTimestamp(`${value.date} ${value.time}`);
  } catch {
    return null;
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
  applyPublishedReleaseTimestamp as a,
  loadPublishedReleaseTimestamp as l
};
