import requests
from bs4 import BeautifulSoup
import re
import ast
import time
import os
from PIL import Image
from io import BytesIO

BASE_URL = "https://dk-sis.hust.edu.vn"
LOGIN_URL = BASE_URL + "/Users/Login.aspx"

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0"
})

os.makedirs("./Captcha/captcha", exist_ok=True)

def add_white_bg_save_jpg_from_bytes(img_bytes, save_path):
    img = Image.open(BytesIO(img_bytes)).convert("RGBA")
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    combined = Image.alpha_composite(bg, img)
    combined = combined.convert("RGB")  # Remove alpha channel
    combined.save(save_path, format="JPEG", quality=95)

with open("./Captcha/private_annote.txt", "w", encoding="utf-8") as annote_file:
    for i in range(10):
        r = session.get(LOGIN_URL)
        soup = BeautifulSoup(r.text, "html.parser")

        viewstate = soup.find("input", {"name": "__VIEWSTATE"})["value"]
        viewstategen = soup.find("input", {"name": "__VIEWSTATEGENERATOR"})["value"]
        eventvalidation = soup.find("input", {"name": "__EVENTVALIDATION"})["value"]

        payload = {
            "__EVENTTARGET": "",
            "__EVENTARGUMENT": "",
            "__VIEWSTATE": viewstate,
            "__VIEWSTATEGENERATOR": viewstategen,
            "__EVENTVALIDATION": eventvalidation,
            "__CALLBACKID": "ccCaptcha",
            "__CALLBACKPARAM": "c0:R"
        }
        r2 = session.post(LOGIN_URL, data=payload)

        match = re.search(r"/\*DX\*/\((.*?)\)$", r2.text)
        if not match:
            print(f"[{i}] Failed to parse DX callback")
            continue

        json_str = match.group(1)

        try:
            data = ast.literal_eval(json_str)
        except Exception as e:
            print(f"[{i}] Failed to parse callback as dict:", e)
            continue

        captcha_path = data.get('result')
        if not captcha_path:
            print(f"[{i}] No captcha URL found in callback")
            continue

        captcha_url = BASE_URL + captcha_path

        img_data = session.get(captcha_url).content
        base_dir = "./Captcha"
        file_path_jpg = f"private/captcha_{i:04d}.jpg"
        add_white_bg_save_jpg_from_bytes(img_data, os.path.join(base_dir,file_path_jpg))

        annote_file.write(f"{file_path_jpg}\n")

        print(f"[{i}] Saved JPG with white background: {file_path_jpg}")
        time.sleep(0.3)

print("Done. 10 images downloaded as JPG with white background.")
