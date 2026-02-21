#!/usr/bin/env python3
import asyncio
import logging
import os
import sys
import signal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

from khors.news_agent.news_agent import NewsAgent, create_default_config


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ]
    )


async def main() -> None:
    telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN') or os.getenv('NEWS_TELEGRAM_BOT_TOKEN')
    telegram_chat_id = os.getenv('NEWS_TELEGRAM_CHAT_ID') or os.getenv('OWNER_CHAT_ID')
    llm_model = os.getenv('NEWS_LLM_MODEL', 'anthropic/claude-3-haiku')

    if not telegram_bot_token:
        print("ERROR: TELEGRAM_BOT_TOKEN not set", file=sys.stderr)
        sys.exit(1)
    if not telegram_chat_id:
        print("ERROR: NEWS_TELEGRAM_CHAT_ID not set", file=sys.stderr)
        sys.exit(1)

    storage_path = os.getenv('NEWS_STORAGE_PATH', 'data/news_agent')
    setup_logging(Path(storage_path) / 'news_agent.log')
    logger = logging.getLogger(__name__)

    config = create_default_config(
        telegram_bot_token=telegram_bot_token,
        telegram_chat_id=telegram_chat_id,
        storage_path=storage_path,
    )
    config.llm_model = llm_model

    agent = NewsAgent(config)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(agent.stop()))

    logger.info(f"Starting news agent. chat_id={telegram_chat_id} model={llm_model}")
    try:
        await agent.run_forever()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())