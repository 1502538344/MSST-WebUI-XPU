#!/usr/bin/env python
# coding: utf-8
"""
MSST WebUI - Gradio-based Web Interface for Music Source Separation

Usage:
    python webui.py [--server_port PORT] [--server_name HOST] [--share] [--lang LANG]
"""

import argparse
import gradio as gr

from webui.inference_tab import create_inference_tab
from webui.training_tab import create_training_tab
from webui.models_tab import create_models_tab
from webui.settings_tab import create_settings_tab
from webui.config import DEFAULT_SERVER_PORT, DEFAULT_SERVER_NAME
from webui.i18n import t, set_lang, get_lang


def create_app():
    """Create the main Gradio application with all tabs."""
    with gr.Blocks(title=t("app_title")) as app:
        gr.Markdown(f"# {t('app_title')}")
        gr.Markdown(t("app_subtitle"))

        with gr.Tabs():
            create_inference_tab()
            create_training_tab()
            create_models_tab()
            create_settings_tab()

        gr.Markdown("---")
        gr.Markdown(f"*{t('powered_by')}*")

    return app


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="MSST WebUI - Music Source Separation")
    parser.add_argument(
        "--server_port",
        type=int,
        default=DEFAULT_SERVER_PORT,
        help=f"Server port (default: {DEFAULT_SERVER_PORT})",
    )
    parser.add_argument(
        "--server_name",
        type=str,
        default=DEFAULT_SERVER_NAME,
        help=f"Server hostname (default: {DEFAULT_SERVER_NAME})",
    )
    parser.add_argument(
        "--share", action="store_true", help="Create a public Gradio link"
    )
    parser.add_argument(
        "--lang",
        type=str,
        default=None,
        choices=["en", "zh"],
        help="Interface language: en (English) or zh (Chinese). Uses saved preference if not specified.",
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Use command line language if provided, otherwise use saved config
    if args.lang:
        set_lang(args.lang)

    current_lang = get_lang()
    lang_name = "English" if current_lang == "en" else "中文"
    print(f"Starting MSST WebUI... (Language: {lang_name})")
    print(f"Server: http://{args.server_name}:{args.server_port}")

    app = create_app()
    app.launch(
        server_name=args.server_name, server_port=args.server_port, share=args.share
    )


if __name__ == "__main__":
    main()
