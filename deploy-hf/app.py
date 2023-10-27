from __future__ import annotations
from typing import Iterable
import gradio as gr
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes
import subprocess
import psutil
from check_sources import construct_prompt_to_use_source

from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from llama_cpp import LlamaRAMCache

hf_hub_download(
    repo_id="szymonrucinski/krakowiak-v2-7b-gguf",
    filename="krakowiak-v2-7b-gguf.Q4_1.bin",
    local_dir=".",
)

llm = Llama(model_path="./krakowiak-v2-7b-gguf.Q4_1.bin", rms_norm_eps=1e-5, n_ctx=2048)
USER_TAG = "### Użytkownik: "
ASSISTANT_TAG = "### Asystent: "
# cache = LlamaRAMCache(capacity_bytes=2 << 30)

# llm.set_cache(cache)

theme = gr.themes.Monochrome(
    primary_hue="orange",
    secondary_hue="red",
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_sm,
    font=[
        gr.themes.GoogleFont("Open Sans"),
        "ui-sans-serif",
        "system-ui",
        "sans-serif",
    ],
)


def get_system_memory():
    memory = psutil.virtual_memory()
    memory_percent = memory.percent
    memory_used = memory.used / (1024.0**3)
    memory_total = memory.total / (1024.0**3)
    return {
        "percent": f"{memory_percent}%",
        "used": f"{memory_used:.3f}GB",
        "total": f"{memory_total:.3f}GB",
    }


def generate(
    instruction,
    max_new_tokens,
    temp,
    top_p,
    top_k,
    rep_penalty,
    enable_internet_search,
):
    if enable_internet_search:
        prompt = construct_prompt_to_use_source(USER_TAG, ASSISTANT_TAG, instruction)
    else:
        # prompt = f"{USER_TAG} {instruction} {ASSISTANT_TAG}"
        prompt = f"<s>[INST] {instruction} [/INST]"

    print(prompt)

    result = ""
    for x in llm(
        prompt,
        stop=[ASSISTANT_TAG],
        stream=True,
        max_tokens=max_new_tokens,
        temperature=temp,
        top_p=top_p,
        top_k=top_k,
        repeat_penalty=rep_penalty,
    ):
        result += x["choices"][0]["text"]
        yield result


examples = [
    "Jaki obiektyw jest idealny do portretów?",
    "Kiedy powinienem wybrać rower gravelowy a kiedy szosowy?",
    "Czym jest sztuczna inteligencja?",
    "Jakie są największe wyzwania sztucznej inteligencji?",
    "Napisz proszę co należy zjeść po ciezkim treningu?",
    "Mam zamiar aplikować na stanowisko menadżera w firmie. Sformatuj mój życiorys.",
]


def process_example(input):
    for x in generate(input, 256, 0.5, 0.9, 40, 1.2):
        pass
    return x


css = ".generating {visibility: hidden} \n footer {visibility: hidden}"


# Based on the gradio theming guide and borrowed from https://huggingface.co/spaces/shivi/dolly-v2-demo
class SeafoamCustom(Base):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.emerald,
        secondary_hue: colors.Color | str = colors.blue,
        neutral_hue: colors.Color | str = colors.blue,
        spacing_size: sizes.Size | str = sizes.spacing_md,
        radius_size: sizes.Size | str = sizes.radius_md,
        font: fonts.Font
        | str
        | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Quicksand"),
            "ui-sans-serif",
            "sans-serif",
        ),
        font_mono: fonts.Font
        | str
        | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            button_primary_background_fill="linear-gradient(90deg, *primary_300, *secondary_400)",
            button_primary_background_fill_hover="linear-gradient(90deg, *primary_200, *secondary_300)",
            button_primary_text_color="white",
            button_primary_background_fill_dark="linear-gradient(90deg, *primary_600, *secondary_800)",
            block_shadow="*shadow_drop_lg",
            button_shadow="*shadow_drop_lg",
            input_background_fill="zinc",
            input_border_color="*secondary_300",
            input_shadow="*shadow_drop",
            input_shadow_focus="*shadow_drop_lg",
        )


seafoam = SeafoamCustom()

with gr.Blocks(theme=seafoam, analytics_enabled=False, css=css) as demo:
    with gr.Column():
        gr.Markdown(
            """ ## 🤖 Krakowiak - Polski model językowy 🤖 \n
                ### by [Szymon Ruciński](https://www.szymonrucinski.pl/) \n
                Wpisz w poniższe pole i kliknij przycisk, aby wygenerować odpowiedzi na najbardziej nurtujące Cię pytania! 🤗 \n
                ***W celu zapewnienia optymalnej wydajności korzystasz z modelu o zredukowanej liczbie parametrów. Jest on 4 razy mniejszy niż oryginalny i generuje odpowiedzi o znacząco niższej jakości.***
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                instruction = gr.Textbox(
                    placeholder="Tutaj wpisz swoje zapytanie.",
                    label="Pytanie",
                    elem_id="q-input",
                )

                with gr.Box():
                    gr.Markdown("**Odpowiedź**")
                    output = gr.Markdown(elem_id="q-output")
                with gr.Row():
                    submit = gr.Button("Wyślij", variant="primary")
                    CHECK_BOX = gr.Checkbox(
                        label="Wyszukaj odpowiedzi w internecie! 🌏 (beta)",
                        variant="primary",
                    )

                with gr.Accordion(label="Zaawansowane Ustawienia", open=False):
                    MAX_NEW_TOKENS = gr.Slider(
                        label="Maksymalna liczba nowych tokenów",
                        minimum=64,
                        maximum=512,
                        step=32,
                        value=512,
                        interactive=True,
                    )
                    TEMP = gr.Slider(
                        label="Stopień Kreatywności",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.1,
                        value=0.8,
                        interactive=True,
                    )
                    TOP_P = gr.Slider(
                        label="Top-P",
                        minimum=0.05,
                        maximum=1.0,
                        step=0.05,
                        value=0.95,
                        interactive=True,
                    )
                    TOP_K = gr.Slider(
                        label="Top-K",
                        minimum=0,
                        maximum=1000,
                        step=1,
                        value=0,
                        interactive=True,
                    )
                    REP_PENALTY = gr.Slider(
                        label="Repetition penalty",
                        minimum=1.0,
                        maximum=2.0,
                        step=0.05,
                        value=1.1,
                        interactive=True,
                    )
                gr.Examples(
                    label="Przykłady",
                    examples=examples,
                    inputs=[instruction],
                    cache_examples=False,
                    fn=process_example,
                    outputs=[output],
                )
                gr.JSON(get_system_memory, every=1)

    click = submit.click(
        generate,
        inputs=[
            instruction,
            MAX_NEW_TOKENS,
            TEMP,
            TOP_P,
            TOP_K,
            REP_PENALTY,
            CHECK_BOX,
        ],
        outputs=[output],
    )
    instruction.submit(
        generate,
        inputs=[
            instruction,
            MAX_NEW_TOKENS,
            TEMP,
            TOP_P,
            TOP_K,
            REP_PENALTY,
            CHECK_BOX,
        ],
        outputs=[output],
    )
# demo.queue(concurrency_count=1, max_size=1).launch(debug=True)


if __name__ == "__main__":
    demo.queue(concurrency_count=1, max_size=1)
    demo.launch(server_name="0.0.0.0", server_port=7860, debug=True)
