import re


def remove_slack_syntax(text):
    # Loại bỏ dấu *
    text = re.sub(r'\*', '', text)

    # Loại bỏ dấu _
    text = re.sub(r'_', '', text)

    # Loại bỏ dấu `
    text = re.sub(r'`', '', text)

    # Loại bỏ link text
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1', text)

    # Loại bỏ code block được bao quanh bởi ba dấu backtick trên và dưới
    text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)

    # Loại bỏ code block được bao quanh bởi ba dấu backtick trên một dòng
    text = re.sub(r'```[^`]*\n', '', text)

    return text


def test():
    # Sử dụng hàm để loại bỏ các kiểu syntax của Slack
    input_text = "```\nHello \n World\n```"
    clean_text = remove_slack_syntax(input_text)
    print(clean_text)  # Kết quả: Hello world code Link


if __name__ == '__main__':
    test()
