import shutil
import subprocess

import torch
import gradio as gr
from fastapi import FastAPI
import os
from PIL import Image
import tempfile
from decord import VideoReader, cpu
from transformers import TextStreamer

from videollava.constants import DEFAULT_IMAGE_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle, Conversation
from utils import Chat, tos_markdown, learn_more_markdown, title_markdown, block_css

import numpy as np



def save_image_to_local(image):
    filename = os.path.join('temp', next(tempfile._get_candidate_names()) + '.jpg')
    image = Image.open(image)
    image.save(filename)
    return filename


def save_video_to_local(video_path):
    filename = os.path.join('temp', next(tempfile._get_candidate_names()) + '.mp4')
    shutil.copyfile(video_path, filename)
    return filename


def generate(image1, video, textbox_in, first_run, state, state_, images_tensor):
    flag = 1
    if not textbox_in:
        if len(state_.messages) > 0:
            textbox_in = state_.messages[-1][1]
            state_.messages.pop(-1)
            flag = 0
        else:
            return "Please enter instruction"

    image1 = image1 if image1 else "none"
    video = video if video else "none"
    # assert not (os.path.exists(image1) and os.path.exists(video))

    if type(state) is not Conversation:
        state = conv_templates[conv_mode].copy()
        state_ = conv_templates[conv_mode].copy()
        images_tensor = []

    first_run = False if len(state.messages) > 0 else True

    text_en_in = textbox_in.replace("picture", "image")

    # images_tensor = [[], []]
    image_processor = handler.image_processor
    if os.path.exists(image1) and not os.path.exists(video):
        tensor = image_processor.preprocess(image1, return_tensors='pt')['pixel_values'][0]
        tensor = tensor.to(handler.model.device, dtype=dtype)
        images_tensor.append(tensor)
    video_processor = handler.video_processor
    if not os.path.exists(image1) and os.path.exists(video):
        tensor = video_processor(video, return_tensors='pt')['pixel_values'][0]
        tensor = tensor.to(handler.model.device, dtype=dtype)
        images_tensor.append(tensor)
    if os.path.exists(image1) and os.path.exists(video):
        tensor = video_processor(video, return_tensors='pt')['pixel_values'][0]
        tensor = tensor.to(handler.model.device, dtype=dtype)
        images_tensor.append(tensor)

        tensor = image_processor.preprocess(image1, return_tensors='pt')['pixel_values'][0]
        tensor = tensor.to(handler.model.device, dtype=dtype)
        images_tensor.append(tensor)

    if first_run:
        if os.path.exists(image1) and not os.path.exists(video):
            text_en_in = DEFAULT_IMAGE_TOKEN + '\n' + text_en_in
        if not os.path.exists(image1) and os.path.exists(video):
            text_en_in = ''.join([DEFAULT_IMAGE_TOKEN] * handler.model.get_video_tower().config.num_frames) + '\n' + text_en_in
        if os.path.exists(image1) and os.path.exists(video):
            text_en_in = ''.join([DEFAULT_IMAGE_TOKEN] * handler.model.get_video_tower().config.num_frames) + '\n' + text_en_in + '\n' + DEFAULT_IMAGE_TOKEN

    text_en_out, state_ = handler.generate(images_tensor, text_en_in, first_run=first_run, state=state_)
    state_.messages[-1] = (state_.roles[1], text_en_out)

    text_en_out = text_en_out.split('#')[0]
    textbox_out = text_en_out

    show_images = ""
    if first_run:
        if os.path.exists(image1):
            filename = save_image_to_local(image1)
            show_images += f'<img src="./file={filename}" style="display: inline-block;width: 250px;max-height: 400px;">'
        if os.path.exists(video):
            filename = save_video_to_local(video)
            show_images += f'<video controls playsinline width="500" style="display: inline-block;"  src="./file={filename}"></video>'

    if flag:
        state.append_message(state.roles[0], textbox_in + "\n" + show_images)
    state.append_message(state.roles[1], textbox_out)

    # Visualize the selected frame
    if os.path.exists(video):
        num_frames = 8
        decord_vr = VideoReader(video, ctx=cpu(0))
        duration = len(decord_vr)
        frame_id_list = np.linspace(0, duration-1, num_frames, dtype=int)
        video_data = decord_vr.get_batch(frame_id_list)
        n, h, w, c = video_data.shape
        video_frame = np.zeros((h, n*w, c), dtype=np.uint8)
        for i in range(n):
            video_frame[:, i*w:(i+1)*w, :] = video_data[i].numpy()
    else:
        video_frame = np.zeros((224, 224*8, 3), dtype=np.uint8)


    return (video_frame, state, state_, state.to_gradio_chatbot(), False, gr.update(value=None, interactive=True), images_tensor, gr.update(value=image1 if os.path.exists(image1) else None, interactive=True), gr.update(value=video if os.path.exists(video) else None, interactive=True))


def regenerate(state, state_):
    state.messages.pop(-1)
    state_.messages.pop(-1)
    if len(state.messages) > 0:
        return state, state_, state.to_gradio_chatbot(), False
    return (state, state_, state.to_gradio_chatbot(), True)


def clear_history(state, state_):
    state = conv_templates[conv_mode].copy()
    state_ = conv_templates[conv_mode].copy()
    return (gr.update(value=None, interactive=True),
            gr.update(value=None, interactive=True),
            gr.update(value=None, interactive=True), \
            gr.update(value=None, interactive=True), \
            True, state, state_, state.to_gradio_chatbot(), [])

if __name__ == "__main__":
    conv_mode = "llava_v1"
    model_path = './checkpoints/HolmesVAD-7B'
    model_base = None
    cache_dir = './cache_dir'
    device = 'cuda:0'
    load_8bit = False
    load_4bit = False
    dtype = torch.float16
    handler = Chat(model_path, conv_mode=conv_mode, model_base=model_base, load_8bit=load_8bit, load_4bit=load_4bit, device=device, cache_dir=cache_dir)
    if not os.path.exists("temp"):
        os.makedirs("temp")
    app = FastAPI()


    textbox = gr.Textbox(
        show_label=False, placeholder="Enter text and press ENTER", container=False
    )
    with gr.Blocks(title='Holmes-VAD🚀', theme=gr.themes.Default(primary_hue="blue", secondary_hue="lime"), css=block_css) as demo:
        gr.Markdown(title_markdown)
        state = gr.State()
        state_ = gr.State()
        first_run = gr.State()
        images_tensor = gr.State()

        with gr.Row():
            with gr.Column(scale=3):
                video = gr.Video(label="Input Video")
                image1 = gr.Image(label="Input Image", type="filepath", visible = False)
        
                cur_dir = os.path.dirname(os.path.abspath(__file__))
                gr.Examples(
                examples=[
                    [
                        f"{cur_dir}/examples/vad/RoadAccidents133_x264_270_451.mp4",
                        "Does the video show any signs of abnormal?",
                    ],
                    [
                        f"{cur_dir}/examples/vad/Fighting033_x264_570_841.mp4",
                        "Are there any unexpected or unusual events in the video?"
                    ],
                    [
                        f"{cur_dir}/examples/vad/v=1djrJ0wxlYo__#1_label_B4-0-0_0_759.mp4",
                        "Are there any abnormal events in the video?"
                    ],
                    [
                        f"{cur_dir}/examples/vad/v=JfLYNEsrTew__#1_label_G-0-0_52_431.mp4",
                        "Does the video exhibit any abnormal sequences?"
                    ],
                    [
                        f"{cur_dir}/examples/vad/A.Beautiful.Mind.2001__#01-14-30_01-16-59_label_A_1845_2101.mp4",
                        "Are there any unexpected or unusual events in the video?"
                    ],
                ],
                inputs=[video, textbox],
                )


            with gr.Column(scale=7):
                video_frame = gr.Image(label="Video Frame Preview", type="filepath")
                chatbot = gr.Chatbot(label="Holmes-VAD", bubble_full_width=True).style(height=750)
                with gr.Row():
                    with gr.Column(scale=8):
                        textbox.render()
                    with gr.Column(scale=1, min_width=50):
                        submit_btn = gr.Button(
                            value="Send", variant="primary", interactive=True
                        )
                with gr.Row(elem_id="buttons") as button_row:
                    upvote_btn = gr.Button(value="👍  Upvote", interactive=True)
                    downvote_btn = gr.Button(value="👎  Downvote", interactive=True)
                    flag_btn = gr.Button(value="⚠️  Flag", interactive=True)
                    # stop_btn = gr.Button(value="⏹️  Stop Generation", interactive=False)
                    regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=True)
                    clear_btn = gr.Button(value="🗑️  Clear history", interactive=True)

        gr.Markdown(tos_markdown)
        gr.Markdown(learn_more_markdown)

        submit_btn.click(generate, [image1, video, textbox, first_run, state, state_, images_tensor],
                        [video_frame, state, state_, chatbot, first_run, textbox, images_tensor, image1, video])

        regenerate_btn.click(regenerate, [state, state_], [state, state_, chatbot, first_run]).then(
            generate, [image1, video, textbox, first_run, state, state_, images_tensor], [video_frame, state, state_, chatbot, first_run, textbox, images_tensor, image1, video])

        clear_btn.click(clear_history, [state, state_],
                        [video_frame, image1, video, textbox, first_run, state, state_, chatbot, images_tensor])

    demo.launch(server_port=8000)

