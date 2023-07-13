# used https://github.com/AndreasFischer1985/code-snippets/blob/master/py/LangChain_HuggingFace_examples.py
# for intial set up


def DownloadHFModel(model_name:str, path:str = "~"):
    from transformers import pipeline
    model = pipeline(model=model_name, device="cpu")
    model.save_pretrained(f"{path}/{model_name}")

def LoadLocalHFModel(model_path:str, temp:float=1e-10, task:str="text-generation"):
    from langchain.llms import HuggingFacePipeline
    llm = HuggingFacePipeline.from_model_id(
        model_id=model_path,
        task=task,
        model_kwargs={"temperature":temp}
    )
    return llm


from langchain import PromptTemplate, LLMChain

# DownloadHFModel("google/flan-t5-small", "./models")
llm = LoadLocalHFModel("./models/google/flan-t5-small")

# from langchain import PromptTemplate, LLMChain
# from langchain.llms import HuggingFacePipeline
# llm = HuggingFacePipeline.from_model_id(model_id="./models/flan-t5-large", task="text2text-generation", model_kwargs={"temperature":1e-10})

template = PromptTemplate(input_variables=["input"], template="{input}")
chain = LLMChain(llm=llm, verbose=True, prompt=template)
chain("What is the meaning of life?")
