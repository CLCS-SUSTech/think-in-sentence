from openai import AsyncClient
import asyncio
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LLMClient:
    def __init__(
        self, 
        base_url, 
        api_key, 
        model,
        # 设置并发量
        concurrency=16,
        retries=3,
        retry_delay=1,
    ):
        self.client = AsyncClient(base_url=base_url, api_key=api_key)
        self.model = model
        self.concurrency = concurrency
        self.retries = retries
        self.retry_delay = retry_delay

    async def single_chat(self, messages, **kwargs):
        for attempt in range(self.retries + 1):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **kwargs
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt == self.retries:
                    logging.error(f"request failed after {self.retries} attempts: {e}")
                    raise e
                logging.warning(f"request failed, {self.retry_delay} seconds later will retry attempt {attempt + 1}... error: {e}")
                await asyncio.sleep(self.retry_delay)
    
    async def single_generate(self, prompt, **kwargs):
        for attempt in range(self.retries + 1):
            try:
                response = await self.client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    **kwargs
                )
                return response.choices[0].text.strip()
            except Exception as e:
                if attempt == self.retries:
                    logging.error(f"request failed after {self.retries} attempts: {e}")
                    raise e
                logging.warning(f"request failed, {self.retry_delay} seconds later will retry attempt {attempt + 1}... error: {e}")
                await asyncio.sleep(self.retry_delay)

    async def async_batch_chat(self, messages_list, **kwargs):
        semaphore = asyncio.Semaphore(self.concurrency)

        async def _bounded_chat(messages):
            async with semaphore:
                return await self.single_chat(messages, **kwargs)

        tasks = [_bounded_chat(msgs) for msgs in messages_list]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
    
    async def async_batch_generate(self, prompts, **kwargs):
        semaphore = asyncio.Semaphore(self.concurrency)

        async def _bounded_generate(prompt):
            async with semaphore:
                return await self.single_generate(prompt, **kwargs)

        tasks = [_bounded_generate(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results

    def chat(self, messages, **kwargs):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                raise RuntimeError("please directly use `await client.async_batch_chat(...)`")
        except RuntimeError as e:
            raise e
        return asyncio.run(self.async_batch_chat(messages, **kwargs))

    def generate(self, prompt, **kwargs):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                raise RuntimeError("please directly use `await client.async_batch_generate(...)`")
        except RuntimeError as e:
            raise e
        return asyncio.run(self.async_batch_generate(prompt, **kwargs))