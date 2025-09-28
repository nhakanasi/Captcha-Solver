def generate_captcha_image(text, font_path="times.ttf", img_w=32, img_h=32):
    import random, math
    from PIL import Image, ImageDraw, ImageFont

    colors = ['black', 'blue']
    text_color = random.choice(colors)
    try:
        font = ImageFont.truetype(font_path, 40)
    except OSError:
        font = ImageFont.load_default()

    # Calculate total width by drawing characters individually with slight overlap
    total_w = 0
    char_widths = []
    for char in text:
        char_bbox = ImageDraw.Draw(Image.new('RGB', (1, 1))).textbbox((0, 0), char, font=font)
        char_w = char_bbox[2] - char_bbox[0]
        char_widths.append(char_w)
        total_w += char_w

    overlap = 8  # pixels to overlap each character
    total_w -= (len(text) - 1) * overlap

    text_bbox = ImageDraw.Draw(Image.new('RGB', (1, 1))).textbbox((0, 0), text[0], font=font)
    text_h = text_bbox[3] - text_bbox[1]

    a = 30.0
    b = 50.0
    phase = random.uniform(0, 2 * math.pi)

    margin = int(abs(a) + text_h * 2)
    text_layer_h = img_h + 2 * margin
    text_layer = Image.new('RGBA', (img_w, text_layer_h), (255, 255, 255, 0))
    text_draw = ImageDraw.Draw(text_layer)

    start_x = (img_w - total_w) / 2
    current_x = start_x
    for i, char in enumerate(text):
        char_bbox = text_draw.textbbox((0, 0), char, font=font)
        char_y = (text_layer_h - text_h) / 2 - char_bbox[1]
        text_draw.text((current_x - char_bbox[0], char_y), char, fill=text_color, font=font)
        current_x += char_widths[i] - overlap

    warped_full = Image.new('RGBA', (img_w, text_layer_h), (255, 255, 255, 0))
    for x in range(img_w):
        sin_val = math.sin((x / b) + phase)
        shift = sin_val * a
        for y in range(text_layer_h):
            src_y_float = y - shift
            if 0 <= src_y_float < text_layer_h - 1:
                floor_y = int(src_y_float)
                frac = src_y_float - floor_y
                p1 = text_layer.getpixel((x, floor_y))
                p2 = text_layer.getpixel((x, floor_y + 1))
                pixel = tuple(int(p1[i] * (1 - frac) + p2[i] * frac) for i in range(4))
                warped_full.putpixel((x, y), pixel)

    non_empty_rows = [y for y in range(text_layer_h) if any(warped_full.getpixel((x, y))[3] > 0 for x in range(img_w))]
    top, bottom = (min(non_empty_rows), max(non_empty_rows)) if non_empty_rows else (0, text_layer_h - 1)
    cropped = warped_full.crop((0, top, img_w, bottom + 1))

    if cropped.height > img_h:
        scale_ratio = img_h / cropped.height
        new_w = int(img_w * scale_ratio)
        cropped = cropped.resize((new_w, img_h), Image.LANCZOS)

    offset_y = (img_h - cropped.height) // 2
    offset_x = (img_w - cropped.width) // 2

    centered_layer = Image.new('RGBA', (img_w, img_h), (255, 255, 255, 0))
    centered_layer.paste(cropped, (offset_x, offset_y))

    img = Image.new('RGB', (img_w, img_h), 'white')
    img.paste(centered_layer, (0, 0), centered_layer)

    return img
for j in range(10):
    for i in range(10):
        img = generate_captcha_image(f"{j}")
        img.save(f"./Captcha/templates/{j}/{i}.png") 