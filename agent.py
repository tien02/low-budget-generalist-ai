import config
import requests
import gradio as gr

def get_response(history) -> str:
    history[-1][1] = ""
    llm_api = f"{config.LLM_API}?question={history[-1][0]}"

    with requests.post(llm_api, stream=True) as r:
        for chunk in r.iter_content(1024):
            history[-1][1] += chunk.decode('utf-8')
            yield history


def set_user_response(user_message:str, chat_history:list)->tuple:
    return '', chat_history + [[user_message, None]]


def add_file(history, file):
    history = history + [((file.name,), None)]
    return history


def get_response_from_img(history):
    history[-1][1] = ""

    with open(history[-1][0][0], 'rb') as img_file:
        files = {"img_files": ("image.jpg", img_file, "image/jpeg")}
        response = requests.post(config.CLIP_API, files=files)
    
    result = response.json()['result']['label']
    
    question = f"Tell me some information about {result}?"

    llm_api = f"{config.LLM_API}?question={question}"

    with requests.post(llm_api, stream=True) as r:
        for chunk in r.iter_content(1024):
            history[-1][1] += chunk.decode('utf-8')
            yield history


if __name__ == '__main__':
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot(label='Medical Assisstance')
        with gr.Row():
            with gr.Column(scale=7):
                msg = gr.Textbox(show_label=False, placeholder='Give some commands')
            with gr.Column(scale=2):
                btn = gr.UploadButton(label="🖼️",file_types=["image"])
            with gr.Column(scale=1):
                clear = gr.Button("Clear")

        msg.submit(set_user_response, [msg, chatbot], [msg, chatbot], queue=False).then(
        get_response, chatbot, chatbot
        )
        btn.upload(add_file, [chatbot, btn], [chatbot], queue=False).then(
        get_response_from_img, chatbot, chatbot)
        clear.click(lambda: None, None, chatbot, queue=False)

    demo.launch(server_name="0.0.0.0", server_port=config.AGENT_PORT)