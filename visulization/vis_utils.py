import random
import numpy as np
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import cv2

def rand_color():
    return (random.randint(0,255), random.randint(0,255), random.randint(0,255))


def draw_bounding_box_on_image(image,
                               boxes,
                               str_list,
                               colors=None,
                               thickness=2,
                               font_file='visulization/RobotoMono-MediumItalic.ttf'):
    #default_color = (0, 250, 154)
    default_color = (255,165,0)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_file)

    for i in range(len(boxes)):
        if colors is not None:
            color = colors[i]
        else:
            color = default_color
        ymin, xmin, ymax, xmax = boxes[i]
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
        draw.line([(left, top), (left, bottom), (right, bottom),
                   (right, top), (left, top)], width=thickness, fill=color)

        dstr = str_list[i]

        display_str_heights = font.getsize(dstr)[1]
        total_display_str_height = (1 + 2 * 0.05) * display_str_heights

        if top > total_display_str_height:
            text_bottom = top
        else:
            text_bottom = bottom + total_display_str_height

        text_width, text_height = font.getsize(dstr)
        margin = np.ceil(0.1 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin), (left + text_width+3*margin, text_bottom)],fill=color)
        if not dstr == '':
            draw.text((left + margin, text_bottom - text_height - margin), dstr, fill='black', font=font)


def draw_label_on_image(image, label_str, pos, size, color, font_file='visulization/RobotoMono-MediumItalic.ttf'):
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_file, size=size)
    left, bottom = pos
    text_width, text_height = font.getsize(label_str)
    margin = np.ceil(0.1 * text_height)
    draw.rectangle([(left, bottom - text_height - 2 * margin), (left + text_width + 3 * margin, bottom)], fill=color)
    draw.text((left + margin, bottom - text_height - margin), label_str, fill='black', font=font)

def draw_label_cv2(cv2_img, label, pos, size, color=None):
    if color == None:
        _color = (255,165,0)
    else:
        _color = color

    cv2_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(cv2_rgb)
    draw_label_on_image(image, label, pos, size, _color)
    cv2_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return cv2_bgr

def draw_from_cv2(cv2_img, boxes, strs, colors=None, thickness=2):
    cv2_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(cv2_rgb)
    draw_bounding_box_on_image(image, boxes, strs, colors, thickness)
    cv2_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return cv2_bgr

def draw_box_img(img_path, boxes, scores, classes, colors=None, output_path=None, show=False):
    img = Image.open(img_path)
    N = len(scores)
    strs = []
    for i in range(N):
        c = classes[i]
        score = scores[i]
        strs.append('%s'%(c))
    draw_bounding_box_on_image(img, boxes, strs, colors)
    if show:
        img.show()
    if output_path is not None:
        img.save(output_path)




def draw_without_str(img_path, boxes, colors=None, output_path=None, show=False):
    img = Image.open(img_path)
    N = len(boxes)
    strs = ['' for _ in range(N)]
    draw_bounding_box_on_image(img, boxes, strs, colors)
    if show:
        img.show()
    if output_path is not None:
        img.save(output_path)




def draw_test():
    f = open('/Users/yangchenhongyi/Documents/TEMP/temp_result/006379.txt', 'r')
    lines = f.readlines()
    bboxes = []
    classes = []
    scores = []
    for line in lines:
        line = line.strip().split(' ')
        if len(line) == 6 or len(line) == 7:
            c = 'Car'
            ymin = int(line[1])
            xmin = int(line[2])
            ymax = int(line[3])
            xmax = int(line[4])
            score = float(line[5])

            classes.append(c)
            bboxes.append([ymin,xmin,ymax,xmax])
            scores.append(score)
    img_path = '/Users/yangchenhongyi/Downloads/kitti_eval/eval_img/006379.png'
    draw_box_img(img_path,bboxes,scores,classes,show=True)



if __name__ == '__main__':
    draw_test()






















