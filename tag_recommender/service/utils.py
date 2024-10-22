import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import httpx
import pandas as pd


async def send_request(client: httpx.AsyncClient, payload):
    """Function to perform a POST request to the recommender API."""
    start_time = time.time()
    try:
        response = await client.post("http://localhost:8000/recommend", json=payload)
        elapsed_time = (time.time() - start_time) * 1000
        return response.status_code, elapsed_time
    except Exception as e:
        elapsed_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        return f"error: {str(e)}", elapsed_time


# Worker function to perform requests
async def worker(
    total_requests: int, results: list, delay: float, tags_list: list[str]
):
    async with httpx.AsyncClient() as client:
        for _ in range(total_requests):
            payload = {"tags": tags_list[_ % len(tags_list)], "num_tags": 30}
            result, elapsed_time = await send_request(client, payload)
            results.append((result, elapsed_time))
            await asyncio.sleep(delay)


def run_stress_test(path: str, workers, requests, rate):
    if not path.endswith(".parquet"):
        raise ValueError("Input file must be a parquet file with a `tags` column.")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    print(f"Loading test tags from {path}...")
    # Load test tags from parquet file
    df = pd.read_parquet(path, columns=["tags"])
    tags_list = df["tags"].apply(lambda arr: ",".join(arr)).tolist()

    delay = 1.0 / rate  # Delay between requests to achieve the desired rate
    total_requests_per_worker = requests // workers
    remaining_requests = requests % workers

    results = []

    print(
        f"Starting stress test with {workers} workers, "
        f"{requests} total requests at {rate} Hz."
    )

    start_time = time.time()

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        tasks = [
            loop.run_in_executor(
                executor,
                asyncio.run,
                worker(total_requests_per_worker, results, delay, tags_list),
            )
            for _ in range(workers)
        ]

        if remaining_requests > 0:
            tasks.append(
                loop.run_in_executor(
                    executor,
                    asyncio.run,
                    worker(remaining_requests, results, delay, tags_list),
                )
            )

        loop.run_until_complete(asyncio.gather(*tasks))

    end_time = time.time()

    # Extract status codes and response times
    status_codes = [r[0] for r in results]
    response_times = [r[1] for r in results]

    # Calculate statistics
    total_time = end_time - start_time
    success_count = sum([1 for r in status_codes if r == 200])
    failure_count = sum([1 for r in status_codes if r != 200])

    response_times = pd.Series(response_times)
    times_statistics = response_times.describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99])

    print(f"Total time taken: {total_time:.2f} seconds")
    print(f"Requests per second: {requests / total_time:.2f} RPS")
    print(f"Successful requests: {success_count}")
    print(f"Failed requests: {failure_count}")

    # Display time statistics in milliseconds
    print("\nResponse time statistics:")
    print(times_statistics)
