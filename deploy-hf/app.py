from __future__ import annotations
from typing import Iterable
import gradio as gr
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes
import subprocess

from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from llama_cpp import LlamaRAMCache

hf_hub_download(
    repo_id="szymonrucinski/krakowiak-7b-gguf",
    filename="krakowiak-7b.gguf.q4_k_m.bin",
    local_dir=".",
)

llm = Llama(model_path="./krakowiak-7b.gguf.q4_k_m.bin", rms_norm_eps=1e-5)

cache = LlamaRAMCache(capacity_bytes=2 << 30)

llm.set_cache(cache)


ins = """### Użytkownik: {} ### Asystent: """

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


def generate(instruction):
    result = ""
    for x in llm(
        ins.format(instruction),
        stop=["### Asystent:"],
        stream=True,
        max_tokens=128,
        temperature=0.5,
    ):
        result += x["choices"][0]["text"]
        yield result


examples = [
    "Jaki obiektyw jest idealny do portretów?",
    "Czym jest sztuczna inteligencja?",
    "Jakie są największe wyzwania sztucznej inteligencji?",
    "Napisz proszę co należy zjeść po ciezkim treningu?",
    "Mam zamiar aplikować na stanowisko menadżera w firmie. Sformatuj mój życiorys.",
]


def process_example(args):
    for x in generate(args):
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
            """ ## Krakowiak Polski model językowy <link href="data:image/x-icon;base64,AAABAAEAEBAAAAAAAABoBQAAFgAAACgAAAAQAAAAIAAAAAEACAAAAAAAAAEAAAAAAAAAAAAAAAEAAAAAAAAsEKkA+fn5ADwT2gA8Fd0Arq6uADoS1QD39/cALw+kAE85mAA6FNgASTOZAPX19QA6E9MA8/PzAPn6+AD8+fsAqKioAHNroACqqKgA/P/7AE05mgAwD6YALQ6pADkQ3QD+/v4AOhXPAPz6+QD8/PwAMhG6ADkR2wDPz88AqKu6AP/+/wA7FNsA5eTmAPr6+gCvrawAMxHDAK+vrwArD6gAMwrXAPr4+ADa2toAOxTcADgV0QDn5+cATjiXAHRrowAzD7kAORPXAP38/gD09PQA///+AOXl5QBQPZoA5eToADQRxQDU1NQANArZADEQpwD9//8ALQm7AP///wD9+/oAORTWACsHuQD9/f0AORTOAMHBwQA8FdwA+/v7ALCurQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAID4YNBM3LxQULzcTNBg+ID48IA42FjAcHDAWNg4gPD4YIB8HOCxACQlALDgHHyAYPCk7JQwJKysrKwkMJRUpPDIuOAUhISsrKyshIQU4CDIiJxkCISsrKysrK0UCQwAiETAxAwMrAwMDAysDAzEwEQo9KDoXHR0dHR0dFzooQQoSKho+Pj4+Pj4+Pj4+PyoSJDkPPj4+Pj4+Pj4+Pg85RzVEBkI+Pj4+Pj4+PkIGRDVCEA1GGD4+Pj4+PhhGDRBCPgYeCxsYPj4+PhgbCx4GPj4+RDUBG0IYGEIbATVEPj4+Pj4BJjNGIyNGMyYBPj4+GD4+PkItBBAQBC1CPj4+GAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=" rel="icon" type="image/x-icon">
                ### by [Szymon Ruciński](https://www.szymonrucinski.pl/) \n
                Wpisz w poniższe pole i kliknij przycisk, aby wygenerować odpowiedzi na najbardziej nurtujące Cię pytania! 😂 \n
                W celu zapewnienia optymalnej wydajności korzystasz z modelu o zredukowanej liczbie parametrów.
            
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
                submit = gr.Button("Wyślij", variant="primary")
                gr.Examples(
                    label="Przykłady",
                    examples=examples,
                    inputs=[instruction],
                    cache_examples=True,
                    fn=process_example,
                    outputs=[output],
                )

    submit.click(generate, inputs=[instruction], outputs=[output])
    instruction.submit(generate, inputs=[instruction], outputs=[output])

# demo.queue(concurrency_count=1).launch(debug=False)
# demo.queue(concurrency_count=1, max_size=20, api_open=False)
# demo.launch(enable_queue=True, share=False)
demo.queue(concurrency_count=1, max_size=3)
demo.launch(debug=True, enable_queue=True, share=False)
