"""
Settings Tab - Application Settings Interface
"""

import os
import json
import gradio as gr
from webui.i18n import t, get_lang, set_lang

SETTINGS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), ".webui_config.json"
)

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")


def load_settings():
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def save_settings(settings):
    try:
        existing = load_settings()
        existing.update(settings)
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def get_available_devices() -> list:
    """Get list of available devices."""
    devices = ["CPU"]
    try:
        import torch

        try:
            import intel_extension_for_pytorch as ipex

            if torch.xpu.is_available():
                count = torch.xpu.device_count()
                for i in range(count):
                    name = torch.xpu.get_device_name(i)
                    devices.insert(0, f"XPU:{i} ({name})")
        except (ImportError, AttributeError):
            pass
    except ImportError:
        pass
    devices.insert(0, "Auto")
    return devices


def get_device_info() -> str:
    """Get current device information."""
    try:
        import torch

        try:
            import intel_extension_for_pytorch as ipex

            if torch.xpu.is_available():
                device_name = torch.xpu.get_device_name(0)
                return f"Intel XPU: {device_name}"
        except (ImportError, AttributeError):
            pass
        return "CPU"
    except ImportError:
        return "CPU (PyTorch not installed)"


def get_system_info() -> dict:
    """Get system information."""
    import sys

    info = {
        "device": get_device_info(),
        "python_version": sys.version.split()[0],
        "ipex_available": False,
    }

    try:
        import torch

        info["pytorch_version"] = torch.__version__

        try:
            import intel_extension_for_pytorch as ipex

            info["ipex_available"] = True
            info["ipex_version"] = ipex.__version__
            if torch.xpu.is_available():
                info["xpu_device_count"] = torch.xpu.device_count()
        except ImportError:
            pass
    except ImportError:
        info["pytorch_version"] = "Not installed"

    return info


def get_log_files():
    if not os.path.exists(LOG_DIR):
        return []
    files = []
    for f in os.listdir(LOG_DIR):
        if f.endswith(".log"):
            full_path = os.path.join(LOG_DIR, f)
            mtime = os.path.getmtime(full_path)
            files.append((f, mtime))
    files.sort(key=lambda x: x[1], reverse=True)
    return [f[0] for f in files]


def read_log_file(filename):
    if not filename:
        return t("no_logs")
    filepath = os.path.join(LOG_DIR, filename)
    if not os.path.exists(filepath):
        return t("file_not_found")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading log: {e}"


def delete_log_file(filename):
    if not filename:
        return t("no_logs"), gr.update(choices=get_log_files(), value=None), ""
    filepath = os.path.join(LOG_DIR, filename)
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            new_files = get_log_files()
            return t("log_deleted"), gr.update(choices=new_files, value=None), ""
        except Exception as e:
            return f"Error: {e}", gr.update(), ""
    return t("file_not_found"), gr.update(), ""


def delete_all_logs():
    if not os.path.exists(LOG_DIR):
        return t("no_logs"), gr.update(choices=[], value=None), ""
    count = 0
    for f in os.listdir(LOG_DIR):
        if f.endswith(".log"):
            try:
                os.remove(os.path.join(LOG_DIR, f))
                count += 1
            except Exception:
                pass
    return f"{t('all_logs_deleted')} ({count})", gr.update(choices=[], value=None), ""


def create_settings_tab():
    """Create the settings tab for application configuration."""
    # Load saved settings
    saved = load_settings()
    device_pref = saved.get("preferred_device", "auto")
    saved_output = saved.get("output_dir", "output")
    saved_models = saved.get("models_dir", "checkpoints")
    saved_configs = saved.get("configs_dir", "configs")

    with gr.Tab(t("tab_settings")) as tab:
        gr.Markdown(f"## {t('settings_title')}")
        gr.Markdown(t("settings_desc"))

        with gr.Row():
            with gr.Column():
                gr.Markdown(f"### {t('language')}")

                current_lang = get_lang()
                current_label = "English" if current_lang == "en" else "中文"

                language_selector = gr.Dropdown(
                    label=t("select_language"),
                    choices=["English", "中文"],
                    value=current_label,
                    interactive=True,
                )

                language_note = gr.HTML(
                    value=f"<p style='font-style:italic;color:#666;'>"
                    + (
                        "Select language and restart."
                        if current_lang == "en"
                        else "选择语言后将自动重启。"
                    )
                    + "</p>"
                )

                gr.Markdown(f"### {t('device_settings')}")

                device_info = gr.Textbox(
                    label=t("current_device"),
                    value=get_device_info(),
                    interactive=False,
                )

                available_devices = get_available_devices()
                # Map saved preference to display value
                saved_device_display = ["Auto"]
                if device_pref == "cpu":
                    saved_device_display = ["CPU"]
                elif device_pref == "auto":
                    saved_device_display = ["Auto"]
                elif "," in device_pref:
                    # Multiple XPUs, e.g., "xpu:0,xpu:1"
                    saved_device_display = []
                    for p in device_pref.split(","):
                        p = p.strip()
                        for d in available_devices:
                            if d.lower().startswith(p.lower()):
                                saved_device_display.append(d)
                                break
                elif device_pref.startswith("xpu:"):
                    for d in available_devices:
                        if d.lower().startswith(device_pref.lower()):
                            saved_device_display = [d]
                            break

                device_choice = gr.Dropdown(
                    label=t("preferred_device"),
                    choices=available_devices,
                    value=saved_device_display,
                    multiselect=True,
                    interactive=True,
                )

                gr.Markdown(f"### {t('paths')}")

                output_dir = gr.Textbox(
                    label=t("default_output_dir"),
                    value=saved_output,
                    placeholder="/path/to/output",
                )

                models_dir = gr.Textbox(
                    label=t("models_dir"),
                    value=saved_models,
                    placeholder="/path/to/models",
                )

                configs_dir = gr.Textbox(
                    label=t("configs_dir"),
                    value=saved_configs,
                    placeholder="/path/to/configs",
                )

            with gr.Column():
                gr.Markdown(f"### {t('system_info')}")

                system_info = gr.JSON(
                    label=t("system_details"),
                    value=get_system_info(),
                )

                refresh_info_btn = gr.Button(t("refresh_info"))

                gr.Markdown(f"### {t('save_load')}")

                with gr.Row():
                    save_btn = gr.Button(t("save_settings"), variant="primary")
                    load_btn = gr.Button(t("load_settings"))

                settings_status = gr.Textbox(label=t("status"), interactive=False)

                gr.Markdown("---")
                exit_btn = gr.Button(t("exit"), variant="stop")

        gr.Markdown("---")
        gr.Markdown(f"## {t('tab_logs')}")
        with gr.Row():
            with gr.Column(scale=1):
                log_dropdown = gr.Dropdown(
                    label=t("select_log"),
                    choices=get_log_files(),
                    allow_custom_value=False,
                )
                with gr.Row():
                    view_log_btn = gr.Button(t("view_log"), variant="primary")
                    refresh_logs_btn = gr.Button("🔄", scale=0, min_width=40)
                with gr.Row():
                    delete_log_btn = gr.Button(t("delete_log"), variant="stop")
                    delete_all_btn = gr.Button(t("delete_all_logs"), variant="stop")
                log_status = gr.Textbox(
                    label=t("status"),
                    value="",
                    interactive=False,
                )
            with gr.Column(scale=3):
                log_content = gr.Code(
                    label=t("log_content"),
                    language=None,
                    lines=25,
                    interactive=False,
                )

        refresh_logs_btn.click(
            fn=lambda: gr.update(choices=get_log_files()),
            inputs=[],
            outputs=[log_dropdown],
        )
        view_log_btn.click(
            fn=read_log_file,
            inputs=[log_dropdown],
            outputs=[log_content],
        )
        log_dropdown.change(
            fn=read_log_file,
            inputs=[log_dropdown],
            outputs=[log_content],
        )
        delete_log_btn.click(
            fn=delete_log_file,
            inputs=[log_dropdown],
            outputs=[log_status, log_dropdown, log_content],
        )
        delete_all_btn.click(
            fn=delete_all_logs,
            inputs=[],
            outputs=[log_status, log_dropdown, log_content],
        )

        def on_language_change(lang_label):
            import os
            import sys
            import threading

            if lang_label is None:
                return gr.update()

            lang_code = "en" if lang_label == "English" else "zh"
            current = get_lang()

            if lang_code == current:
                return gr.update()

            set_lang(lang_code)
            msg = "Restarting..." if lang_code == "en" else "正在重启..."

            def do_restart():
                import time

                time.sleep(1.5)
                os.execv(sys.executable, [sys.executable] + sys.argv)

            threading.Thread(target=do_restart, daemon=True).start()

            # Return HTML with embedded iframe that polls for reload
            return f"""
            <p style='font-style:italic;color:#666;'>{msg}</p>
            <iframe style="display:none" srcdoc="
                <script>
                    function poll() {{
                        fetch('{"/"}', {{method: 'HEAD', cache: 'no-store'}})
                            .then(() => parent.location.reload())
                            .catch(() => setTimeout(poll, 500));
                    }}
                    setTimeout(poll, 2000);
                </script>
            "></iframe>
            """

        language_selector.select(
            fn=on_language_change,
            inputs=[language_selector],
            outputs=[language_note],
        )

        def refresh_system_info():
            return get_system_info()

        refresh_info_btn.click(
            fn=refresh_system_info,
            inputs=[],
            outputs=[system_info],
        )

        def do_save_settings(device_prefs, out_dir, mod_dir, cfg_dir):
            # Parse device preferences (now a list)
            if not device_prefs or device_prefs == ["Auto"] or "Auto" in device_prefs:
                device_value = "auto"
            elif device_prefs == ["CPU"] or "CPU" in device_prefs:
                device_value = "cpu"
            else:
                # Multiple XPUs or single XPU
                device_ids = []
                for pref in device_prefs:
                    if pref.startswith("XPU:"):
                        # Extract XPU index, e.g., "XPU:0 (Intel Arc)" -> "xpu:0"
                        device_ids.append(pref.split(" ")[0].lower())
                device_value = ",".join(device_ids) if device_ids else "auto"

            settings = {
                "preferred_device": device_value,
                "output_dir": out_dir,
                "models_dir": mod_dir,
                "configs_dir": cfg_dir,
            }
            if save_settings(settings):
                return t("saved")
            return t("save_failed")

        save_btn.click(
            fn=do_save_settings,
            inputs=[device_choice, output_dir, models_dir, configs_dir],
            outputs=[settings_status],
        )

        def do_load_settings():
            settings = load_settings()
            device_pref = settings.get("preferred_device", "auto")
            available = get_available_devices()

            # Map saved preference to display values (list)
            display_devices = ["Auto"]
            if device_pref == "cpu":
                display_devices = ["CPU"]
            elif device_pref == "auto":
                display_devices = ["Auto"]
            elif "," in device_pref:
                # Multiple XPUs
                display_devices = []
                for p in device_pref.split(","):
                    p = p.strip()
                    for d in available:
                        if d.lower().startswith(p.lower()):
                            display_devices.append(d)
                            break
            elif device_pref.startswith("xpu:"):
                for d in available:
                    if d.lower().startswith(device_pref.lower()):
                        display_devices = [d]
                        break

            return (
                display_devices,
                settings.get("output_dir", "output"),
                settings.get("models_dir", "checkpoints"),
                settings.get("configs_dir", "configs"),
                t("loaded"),
            )

        load_btn.click(
            fn=do_load_settings,
            inputs=[],
            outputs=[
                device_choice,
                output_dir,
                models_dir,
                configs_dir,
                settings_status,
            ],
        )

        def do_exit():
            import os
            import threading

            print("Shutting down WebUI...")

            def force_exit():
                import time

                time.sleep(0.5)
                os._exit(0)

            threading.Thread(target=force_exit, daemon=True).start()
            return "Exiting..."

        exit_btn.click(fn=do_exit, inputs=[], outputs=[settings_status])

    return tab
