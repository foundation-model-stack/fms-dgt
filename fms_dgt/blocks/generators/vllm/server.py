# Standard
from argparse import Namespace
import asyncio

# Third Party
from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.utils import FlexibleArgumentParser
import psutil
import uvloop


async def main(args: Namespace):
    """Runs the vllm server while checking that the original process is still alive"""
    pid, check_interval = args.pid, args.check_interval

    delattr(args, "pid")
    delattr(args, "check_interval")

    monitor_task = monitor(pid, check_interval)
    server_task = run_server(args)

    finished, unfinished = await asyncio.wait(
        [monitor_task, server_task], return_when=asyncio.FIRST_COMPLETED
    )
    for x in finished:
        result = x.result()
        if result:
            # cancel the other tasks, we have a result. We need to wait for the cancellations
            for task in unfinished:
                task.cancel()
            await asyncio.wait(unfinished)
            return result


async def monitor(parent_pid: int, check_interval: float):
    while True:
        await asyncio.sleep(check_interval)
        if not psutil.pid_exists(parent_pid):
            return


if __name__ == "__main__":
    # This section should be in sync with vllm/scripts.py for CLI entrypoints.
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )
    parser = make_arg_parser(parser)
    parser.add_argument("--pid", required=True, type=int)
    parser.add_argument("--check-interval", required=True, type=float)
    args = parser.parse_args()

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    uvloop.run(main(args))
