export async function fetchCSV(url) {
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`Failed to fetch dataset: ${response.status} ${response.statusText}`);
  }

  return await response.text();
}