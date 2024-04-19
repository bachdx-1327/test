import io
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import argparse
import torch
from decouple import config

from utils import remove_slack_syntax
from imggen.diffusion_model import DiffusionGenerationV2
from promptgen.prompt_gen import MagicGeneration

# init params
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SLACK_BOT_TOKEN = config('SLACK_BOT_TOKEN')
SLACK_APP_TOKEN = config('SLACK_APP_TOKEN')

# Initializes Slack app with bot tokens and gen model
app = App(token=SLACK_BOT_TOKEN)
gen_model = None
prompt_model = None

# Init argument
args = None


# Listen and handle slash command for stable diffusion image generation
@app.command("/create-image")
def create_image(ack, command, client):
    # Acknowledge command request from slack
    ack()

    # Get prompt from command text and add midjourney style to it
    prompt = f"Create an Image about {command['text']}"
    prompt = remove_slack_syntax(prompt)
    # Gen Image and convert to binary format
    image = gen_model.generate_image(prompt, width=args.width_image, height=args.height_image)
    image_binary = io.BytesIO()
    image.save(image_binary, format='JPEG')
    image_binary_bytes = image_binary.getvalue()

    # Post message to channel indicating that image is being generated
    initial_message = client.chat_postMessage(channel=command["channel_id"], text="Generating image...")
    client.files_upload_v2(file=image_binary_bytes, filename='Generated_Image.jpg', channels=command["channel_id"])

    client.chat_update(
        channel=command["channel_id"],
        ts=initial_message["ts"],
        blocks=[
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"Image generated! :white_check_mark:. Prompt: {prompt}"
                }
            }
        ]
    )

@app.command("/gen-prompt")
def generate_prompt(ack, command, client):
    # Acknowledge command request from slack
    ack()

    # Get prompt from command text and add midjourney style to it
    prompt = f"Create an Image about {command['text']}"
    prompt = remove_slack_syntax(prompt)

    # Post message to channel indicating that image is being generated
    initial_message = client.chat_postMessage(channel=command["channel_id"], text="Generating prompt...")

    # Get prompt 
    lst_prompt = prompt_model.generate_prompt(prompt)

    client.chat_update(
        channel=command["channel_id"],
        ts=initial_message["ts"],
        blocks=[
            {
			"type": "section",
			"text": {
				"type": "plain_text",
				"text": lst_prompt[0],
			}
		},
		{
			"type": "section",
			"text": {
				"type": "plain_text",
				"text": lst_prompt[1],
			}
		},
		{
			"type": "section",
			"text": {
				"type": "plain_text",
				"text": lst_prompt[2],
			}
		},
		{
			"type": "section",
			"text": {
				"type": "plain_text",
				"text": lst_prompt[3],
			}
		}
        ]
    )



def run():
    global gen_model

    # load model checkpoint from huggingface
    gen_model = DiffusionGenerationV2(device=device, torch_dtype=args.torch_dtype,
                                      num_inference_steps=args.num_inference_steps)
    gen_model.load_checkpoint(checkpoint_name=args.checkpoint_name)

    # Load and get prompt model
    promt_model = MagicGeneration(device=device)
    promt_model.load_checkpoint()

    # start app
    SocketModeHandler(app, SLACK_APP_TOKEN).start()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cn', '--checkpoint-name', type=str, default="stabilityai/stable-diffusion-2-1")
    parser.add_argument('-wi', '--width-image', type=int, default=512)
    parser.add_argument('-hi', '--height-image', type=int, default=512)
    parser.add_argument('-td', '--torch-dtype', type=int, default=16)
    parser.add_argument('-n', '--num_inference_steps', type=int, default=25)
    tmp_args = parser.parse_args()
    global args
    args = tmp_args


if __name__ == "__main__":
    parse_opt()
    run()
