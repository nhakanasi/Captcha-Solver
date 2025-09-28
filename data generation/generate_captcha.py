from PIL import Image, ImageDraw, ImageFont
import string, random
import math
import os

def random_str():
    N = 5
    return ''.join(random.choices(string.digits, k=N))

def get_random_point():
    return (random.randrange(5, 195), random.randrange(5, 75))

colors = ['black', 'blue']
fill_colors = ['black', 'blue', 'white']

def gen_captcha(n):
    try:
        os.makedirs("./Captcha/img", exist_ok=True)  # Create directory if it doesn't exist
    except Exception as e:
        print(f'dir exist or error: {e}')
    with open('./Captcha/annote.txt', 'w') as f:
        for _ in range(n):
            img = Image.new('RGB', (200, 80), 'white')
            draw = ImageDraw.Draw(img)

            captcha = random_str()
            text_color = random.choice(colors)
            font = ImageFont.load_default(size=40)  # Slightly smaller for better fit

            # Enhanced noise: dots, lines, and speckles
            for _ in range(random.randrange(15, 25)):  # More noise dots
                draw.point(get_random_point(), fill=random.choice(fill_colors))
            
            for _ in range(random.randrange(3, 6)):  # Add random lines
                start = get_random_point()
                end = get_random_point()
                draw.line([start, end], fill=random.choice(colors), width=1)
            
            for _ in range(random.randrange(5, 10)):  # Add speckles
                x, y = get_random_point()
                draw.ellipse([x-2, y-2, x+2, y+2], fill=random.choice(fill_colors))

            # Get text bounding box for the whole string
            text_bbox = draw.textbbox((0, 0), captcha, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]

            # Sine parameters: larger amplitude and longer wavelength for bigger, gentler curve
            a = 40.0  # Amplitude (bigger curve)
            b = 100.0  # Wavelength factor (larger for small part of sine curve)

            padding = math.ceil(a)
            text_layer_h = 80 + 2 * padding
            text_layer = Image.new('RGBA', (200, text_layer_h), (255, 255, 255, 0))
            text_draw = ImageDraw.Draw(text_layer)

            # Position text horizontally centered, vertically with padding
            text_x = (200 - text_w) / 2 - text_bbox[0]
            text_y = padding - text_bbox[1]
            text_draw.text((text_x, text_y), captcha, fill=text_color, font=font)

            # Create warped layer by displacing pixels along sine curve
            warped_layer = Image.new('RGBA', (200, 80), (255, 255, 255, 0))
            for x in range(200):
                sin_val = math.sin(x / b)
                shift = sin_val * a
                for y in range(80):
                    src_y_float = y - shift + padding
                    if src_y_float < 0 or src_y_float >= text_layer_h - 1:
                        continue
                    floor_y = math.floor(src_y_float)
                    frac = src_y_float - floor_y
                    p1 = text_layer.getpixel((x, floor_y))
                    p2 = text_layer.getpixel((x, floor_y + 1))
                    pixel = tuple(int(p1[i] * (1 - frac) + p2[i] * frac) for i in range(4))
                    warped_layer.putpixel((x, y), pixel)

            # Paste the warped text onto the main image
            img.paste(warped_layer, (0, 0), warped_layer)

            path = f"./Captcha/img/{captcha}.png"
            img.save(path)
            f.write(f"img/{captcha}.png\t{captcha}\n")

def gen_captcha2(n):
    os.makedirs("./Captcha/img", exist_ok=True)
    with open('./Captcha/annote.txt', 'w') as f:
        for _ in range(n):
            img_w, img_h = 200, 80
            captcha = random_str()
            text_color = random.choice(colors)
            try:
                font = ImageFont.truetype("arial.ttf", 48)
            except OSError:
                font = ImageFont.load_default()

            text_bbox = ImageDraw.Draw(Image.new('RGB', (1, 1))).textbbox((0, 0), captcha, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]

            a = 30.0
            b = 60.0
            phase = random.uniform(0, 2 * math.pi)

            margin = int(abs(a) + text_h * 2)  # extra padding for large shifts
            text_layer_h = img_h + 2 * margin
            text_layer = Image.new('RGBA', (img_w, text_layer_h), (255, 255, 255, 0))
            text_draw = ImageDraw.Draw(text_layer)

            text_x = (img_w - text_w) / 2 - text_bbox[0]
            text_y = (text_layer_h - text_h) / 2 - text_bbox[1]
            text_draw.text((text_x, text_y), captcha, fill=text_color, font=font)

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

            # Get bounding box of non-transparent pixels and center in final image
            non_empty_rows = [y for y in range(text_layer_h) if any(warped_full.getpixel((x, y))[3] > 0 for x in range(img_w))]
            top, bottom = (min(non_empty_rows), max(non_empty_rows)) if non_empty_rows else (0, text_layer_h - 1)
            cropped = warped_full.crop((0, top, img_w, bottom + 1))

            # Ensure cropped height fits
            if cropped.height > img_h:
                scale_ratio = img_h / cropped.height
                new_w = int(img_w * scale_ratio)
                cropped = cropped.resize((new_w, img_h), Image.LANCZOS)

            offset_y = (img_h - cropped.height) // 2
            offset_x = (img_w - cropped.width) // 2

            centered_layer = Image.new('RGBA', (img_w, img_h), (255, 255, 255, 0))
            centered_layer.paste(cropped, (offset_x, offset_y))

            img = Image.new('RGB', (img_w, img_h), 'white')
            draw = ImageDraw.Draw(img)
            for _ in range(random.randrange(15, 25)):
                draw.point(get_random_point(), fill=random.choice(fill_colors))
            for _ in range(random.randrange(3, 6)):
                draw.line([get_random_point(), get_random_point()], fill=random.choice(colors), width=1)
            for _ in range(random.randrange(5, 10)):
                x, y = get_random_point()
                draw.ellipse([x-2, y-2, x+2, y+2], fill=random.choice(fill_colors))

            img.paste(centered_layer, (0, 0), centered_layer)

            path = f"./Captcha/img/{captcha}.png"
            img.save(path)
            f.write(f"img/{captcha}.png\t{captcha}\n")

def gen_captcha3(n):
    os.makedirs("./Captcha/img", exist_ok=True)
    with open('./Captcha/annote.txt', 'w') as f:
        for _ in range(n):
            img_w, img_h = 200, 80
            captcha = random_str()
            text_color = random.choice(colors)
            try:
                font = ImageFont.truetype("times.ttf", 48)
            except OSError:
                font = ImageFont.load_default()

            # Calculate total width by drawing characters individually with slight overlap
            total_w = 0
            char_widths = []
            for char in captcha:
                char_bbox = ImageDraw.Draw(Image.new('RGB', (1, 1))).textbbox((0, 0), char, font=font)
                char_w = char_bbox[2] - char_bbox[0]
                char_widths.append(char_w)
                total_w += char_w
            
            # Reduce total width to eliminate spacing - make characters overlap slightly
            overlap = 8  # pixels to overlap each character
            total_w -= (len(captcha) - 1) * overlap
            
            # Use the total width and height for positioning
            text_bbox = ImageDraw.Draw(Image.new('RGB', (1, 1))).textbbox((0, 0), captcha[0], font=font)
            text_h = text_bbox[3] - text_bbox[1]

            a = 30.0
            b = 50.0
            phase = random.uniform(0, 2 * math.pi)

            margin = int(abs(a) + text_h * 2)
            text_layer_h = img_h + 2 * margin
            text_layer = Image.new('RGBA', (img_w, text_layer_h), (255, 255, 255, 0))
            text_draw = ImageDraw.Draw(text_layer)

            # Draw characters individually with overlap to eliminate spacing
            start_x = (img_w - total_w) / 2
            current_x = start_x
            for i, char in enumerate(captcha):
                char_bbox = text_draw.textbbox((0, 0), char, font=font)
                char_y = (text_layer_h - text_h) / 2 - char_bbox[1]
                text_draw.text((current_x - char_bbox[0], char_y), char, fill=text_color, font=font)
                current_x += char_widths[i] - overlap  # Subtract overlap to make characters touch

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

            # Get bounding box of non-transparent pixels and center in final image
            non_empty_rows = [y for y in range(text_layer_h) if any(warped_full.getpixel((x, y))[3] > 0 for x in range(img_w))]
            top, bottom = (min(non_empty_rows), max(non_empty_rows)) if non_empty_rows else (0, text_layer_h - 1)
            cropped = warped_full.crop((0, top, img_w, bottom + 1))

            # Ensure cropped height fits
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
            draw = ImageDraw.Draw(img)
            for _ in range(random.randrange(15, 25)):
                draw.point(get_random_point(), fill=random.choice(fill_colors))
            for _ in range(random.randrange(3, 6)):
                draw.line([get_random_point(), get_random_point()], fill=random.choice(colors), width=1)
            for _ in range(random.randrange(5, 10)):
                x, y = get_random_point()
                draw.ellipse([x-2, y-2, x+2, y+2], fill=random.choice(fill_colors))

            path = f"./Captcha/img/{captcha}.png"
            img.save(path)
            f.write(f"img/{captcha}.png\t{captcha}\n")

GLOBAL_SCALE_RANGE = (0.85, 1.20)   # Slightly more scale variation
CHAR_OVERLAP = 15                    # Slightly more overlap
WARP_AMPLITUDE = 35               # Bigger curve
WARP_WAVELENGTH = 65.0              # Smoother wave
BOX_PAD_PIX = 0            # Small padding
BOX_PAD_RATIO = 0.0                 # 10% padding ratio
def gen_captcha4(n):
    os.makedirs("./Self-taught/Captcha/img", exist_ok=True)
    with open('./Self-taught/Captcha/annote.txt','w') as annote_file, open('./Self-taught/Captcha/yolo.txt','w') as yolo_file:
        for idx in range(n):
            img_w, img_h = 200, 80
            captcha = random_str()
            color = random.choice(colors)
            font = ImageFont.truetype('timesi.ttf', 70)
            # Measure per character width/height (remove left bearing)
            char_metrics = []
            total_w = 0
            tmp_draw = ImageDraw.Draw(Image.new("RGB",(1,1)))
            for ch in captcha:
                bb = tmp_draw.textbbox((0,0), ch, font=font)  # (l,t,r,b)
                cw = bb[2]-bb[0]
                ch_h = bb[3]-bb[1]
                char_metrics.append((ch, bb, cw, ch_h))
                total_w += cw
            if len(char_metrics) > 1:
                total_w -= (len(char_metrics)-1)*CHAR_OVERLAP

            # Warp setup
            a = WARP_AMPLITUDE
            b = WARP_WAVELENGTH
            phase = random.uniform(0, 2*math.pi)

            max_h = max(h for (_c,_bb,_w,h) in char_metrics)
            margin = int(abs(a) + max_h*2)
            text_layer_h = img_h + 2*margin
            text_layer = Image.new("RGBA",(img_w,text_layer_h),(0,0,0,0))
            draw_layer = ImageDraw.Draw(text_layer)

            # Draw characters (store pre-warp spans)
            spans = []
            current_x = (img_w - total_w)/2
            baseline_y = (text_layer_h - max_h)/2

            for i,(ch, bb, cw, ch_h) in enumerate(char_metrics):
                # Draw each glyph
                gy = baseline_y
                draw_layer.text((current_x - bb[0], gy - bb[1]), ch, font=font, fill=color)
                spans.append((current_x, current_x+cw, gy, gy+ch_h))
                
                if i < len(char_metrics)-1:
                    # Dynamic overlap based on current and next character
                    next_cw = char_metrics[i+1][2]
                    # Use larger overlap for narrow characters like 1, 7, 4
                    dynamic_overlap = max(CHAR_OVERLAP, min(cw, next_cw) * 0.3)
                    current_x += cw - dynamic_overlap

            # Warp (vertical sine)
            warped_full = Image.new("RGBA",(img_w,text_layer_h),(0,0,0,0))
            for x in range(img_w):
                # Add multiple wave components for more complex curve
                shift1 = a * math.sin((x / b) + phase)
                shift2 = (a * 0.3) * math.sin((x / (b * 0.7)) + phase + math.pi/4)  # Secondary wave
                shift = shift1 + shift2  # Combine waves for more complexity
                
                for y in range(text_layer_h):
                    sy = y - shift
                    if 0 <= sy < text_layer_h-1:
                        fy = int(sy)
                        frac = sy - fy
                        p1 = text_layer.getpixel((x,fy))
                        p2 = text_layer.getpixel((x,fy+1))
                        warped_full.putpixel((x,y),
                            tuple(int(p1[c]*(1-frac)+p2[c]*frac) for c in range(4)))


            # Crop vertically to content
            rows = [y for y in range(text_layer_h)
                    if any(warped_full.getpixel((xx,y))[3] > 0 for xx in range(img_w))]
            top, bottom = (min(rows), max(rows)) if rows else (0, text_layer_h-1)
            cropped = warped_full.crop((0, top, img_w, bottom+1))

            # Random global scale (uniform) while keeping inside canvas
            scale = random.uniform(*GLOBAL_SCALE_RANGE)
            # First clamp by height
            if int(cropped.height * scale) > img_h:
                scale = img_h / cropped.height
            # Then clamp by width
            if int(cropped.width * scale) > img_w:
                scale = min(scale, img_w / cropped.width)

            if scale != 1.0:
                new_w = max(1,int(round(cropped.width * scale)))
                new_h = max(1,int(round(cropped.height * scale)))
                cropped = cropped.resize((new_w, new_h), Image.LANCZOS)

            offset_x = (img_w - cropped.width)//2
            offset_y = (img_h - cropped.height)//2

            final_layer = Image.new("RGBA",(img_w,img_h),(0,0,0,0))
            
            final_layer.paste(cropped,(offset_x,offset_y))
            final_img = Image.new("RGB",(img_w,img_h),"white")
            draw = ImageDraw.Draw(final_img)
            for _ in range(random.randrange(15, 25)):
                draw.point(get_random_point(), fill=random.choice(fill_colors))
            for _ in range(random.randrange(3, 6)):
                draw.line([get_random_point(), get_random_point()], fill=random.choice(colors), width=1)
            for _ in range(random.randrange(5, 10)):
                x, y = get_random_point()
                draw.ellipse([x-2, y-2, x+2, y+2], fill=random.choice(fill_colors))
            final_img.paste(final_layer,(0,0),final_layer)


            # Derive per-char warped boxes analytically
            # For each original span (x0,x1,y0,y1):
            boxes = []
            scale_ratio = scale  # single uniform scale from cropped
            # We need scale factor applied to vertical coordinates after (y - top)
            # Effective vertical scale is cropped.height/original_cropped_height; we used 'scale'.
            # For analytic mapping: after warp and crop, prior to scale:
            original_cropped_height = (bottom+1 - top)
            effective_scale = cropped.height / (bottom+1 - top)

            for (x0, x1, y0, y1) in spans:
                # sample shifts along x within glyph span
                xs = range(int(x0), int(x1)+1)
                if not xs:
                    continue
                # Calculate shifts with both wave components
                shifts = []
                for x in xs:
                    shift1 = a * math.sin((x / b) + phase)
                    shift2 = (a * 0.3) * math.sin((x / (b * 0.7)) + phase + math.pi/4)
                    shifts.append(shift1 + shift2)
                
                min_shift = min(shifts)
                max_shift = max(shifts)
                wy0 = (y0 + min_shift) - top
                wy1 = (y1 + max_shift) - top
                # apply scaling
                wy0 *= effective_scale
                wy1 *= effective_scale
                wx0 = (x0) * (cropped.width / img_w)  # horizontal scale preserves aspect (cropped.width==img_w usually)
                wx1 = (x1) * (cropped.width / img_w)

                # Final offsets
                fx0 = int(round(wx0 + offset_x))
                fx1 = int(round(wx1 + offset_x))
                fy0 = int(round(wy0 + offset_y))
                fy1 = int(round(wy1 + offset_y))

                # Clamp
                fx0 = max(0,min(img_w-1,fx0))
                fx1 = max(0,min(img_w-1,fx1))
                fy0 = max(0,min(img_h-1,fy0))
                fy1 = max(0,min(img_h-1,fy1))
                if fx1 > fx0 and fy1 > fy0:
                    boxes.append((fx0, fy0, fx1, fy1))

            # YOLO line
            padded_boxes = []
            for (x0,y0,x1,y1) in boxes:
                w = x1 - x0
                h = y1 - y0
                pad_x = max(BOX_PAD_PIX, int(round(w * BOX_PAD_RATIO)))
                pad_y = max(BOX_PAD_PIX, int(round(h * BOX_PAD_RATIO)))
                nx0 = max(0, x0 - pad_x)
                ny0 = max(0, y0 - pad_y)
                nx1 = min(img_w-1, x1 + pad_x)
                ny1 = min(img_h-1, y1 + pad_y)
                # Ensure still valid
                if nx1 > nx0 and ny1 > ny0:
                    padded_boxes.append((nx0, ny0, nx1, ny1))
            boxes = padded_boxes  # replace with padded

            # Save
            filename = f"{idx}_{captcha}.png"
            final_img.save(f"./Self-taught/Captcha/img/{filename}")
            annote_file.write(f"img/{filename}\t{captcha}\n")

            # YOLO line (using padded boxes)
            yline = f"img/{filename}"
            for i, (x0,y0,x1,y1) in enumerate(boxes):
                w = x1 - x0
                h = y1 - y0
                cx = (x0 + w/2)/img_w
                cy = (y0 + h/2)/img_h
                bw = w/img_w
                bh = h/img_h
                yline += f" {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {int(captcha[i])}"
            yolo_file.write(yline + "\n")

            if idx == 0:
                dbg = final_img.copy()
                d = ImageDraw.Draw(dbg)
                for (x0,y0,x1,y1) in boxes:
                    d.rectangle([x0,y0,x1,y1], outline="red", width=2)
                dbg.save("./Self-taught/Captcha/debug_first.png")
                print(f"[INFO] debug_first.png boxes={len(boxes)} scale={scale:.3f} padding(px,ratio)=({BOX_PAD_PIX},{BOX_PAD_RATIO})")


# Call
gen_captcha4(400)