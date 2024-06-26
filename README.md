# Stable Diffusion PoC

git clone repos: 
```
git clone https://github.com/anhalu/slack-bot-duffusion-model.git
```

create .env file and setup token
```
SLACK_BOT_TOKEN = "<your-bot-token>"
SLACK_APP_TOKEN = "<your-app-token>"
```

create new env conda & install requirements.txt 
```
conda create -n slack-bot python=3.8
conda activate slack-bot
pip install -r requirements.txt
```

run file main.py
```
python app/app.py
```

you can modify parameters in parameters.py or just parse parameter in command line
```
python app/app.py --checkpoint-name <name-in-huggingface> --width_image <width> --height_image <height> --torch-dtype <16 or 32>
```

or can run repos by docker 
```
./run_docker.sh
```