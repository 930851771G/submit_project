from pathlib import Path
from typing import Annotated, Union

import typer
from peft import AutoPeftModelForCausalLM, PeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
#导入模块和方法

#ModelType 和 TokenizerType 都是 Union，它们可以接受一个或多个类型作为其成员。
ModelType = Union[PreTrainedModel, PeftModelForCausalLM]
TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

#这是一个创建 typer 对象的传统方式，用于创建一个具有特定选项的 Python Typer 实例。
app = typer.Typer(pretty_exceptions_show_locals=False)


#_resolve_path函数：将一个相对路径转换为完全路径。
def _resolve_path(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve()

def load_model_and_tokenizer(model_dir: Union[str, Path]) -> tuple[ModelType, TokenizerType]:
    model_dir = _resolve_path(model_dir)
    if (model_dir / 'adapter_config.json').exists():
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=True, device_map='auto'
        )
        tokenizer_dir = model.peft_config['default'].base_model_name_or_path
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=True, device_map='auto'
        )
        tokenizer_dir = model_dir
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir, trust_remote_code=True
    )
    return model, tokenizer


@app.command()
def main(
        model_dir: Annotated[str, typer.Argument(help='')],
        out_dir: Annotated[str, typer.Option(help='')],
):
    model, tokenizer = load_model_and_tokenizer(model_dir)
    
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)

if __name__ == '__main__':
    app()