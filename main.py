from PIL import Image
import sys
import re
from pdf2image import convert_from_path
import pyocr
import pyocr.builders
import itertools
import numpy as np
import pandas as pd
import cv2


def init_ocr(path_engine):
    """
    Initialise the OCR engine
    """
    pyocr.tesseract.TESSERACT_CMD = path_engine
    tools = pyocr.get_available_tools()
    if len(tools) == 0:
        print("No OCR tool found")
        sys.exit(1)

    tool = tools[0]
    print("Will use tool '%s'" % (tool.get_name()))

    langs = tool.get_available_languages()
    print("Available languages: %s" % ", ".join(langs))
    lang = 'rus'
    print("Will use lang '%s'" % (lang))
    return tool, lang


def get_text_from_images(images, lang, builder):
    """
    Get the text from the images
    """
    if len(images) == 0:
        print("No images found")
        sys.exit(1)

    text = ''
    for image in images:
        text += tool.image_to_string(image, lang=lang, builder=builder)
    return text


def check_orientation(image, lang):
    """
    Check the orientation of the image
    """
    if tool.can_detect_orientation():
        try:
            orientation = tool.detect_orientation(
                image,
                lang=lang
            )
        except pyocr.PyocrException as exc:
            print("Orientation detection failed: {}".format(exc))
            return
        print("Orientation: {}".format(orientation))


def check_count_scip_words(a, len_key, func_sorting, skip):
    """
    Check the number of scip words in a string
    """
    if (len(a) != len_key) or not (func_sorting(np.array((a)))):
        return False, a[0]
    elif skip >= sum([(x - y - 1) for x, y in zip(a[::-1], a[::-1][1:])]):
        return True, a[0]
    return False, a[0]


def find_sub_with_skip(text, key, skip, func_sorting):
    """
    Find the first sub-string in the text
    """
    text = text.lower().split()
    key = key.lower().split()
    indexes = []
    for i in key:
        indices = [j for j in range(len(text)) if text[j] == i]
        indexes.append(indices)
    indexes = list(itertools.product(*indexes))
    rezult = [check_count_scip_words(i, len(key), func_sorting, skip) for i in indexes]
    is_normal_skip = [i[0] for i in rezult]
    if sum(is_normal_skip) > 0:
        index_start_position = rezult[is_normal_skip.index(True)]
        return True, index_start_position
    return False, 0


def prepare_text(text):
    text = text.replace('\n', ' ')
    text = re.sub(r'[a-zA-Z0-9]', '', text)
    return " ".join(text.split())


def get_keyphrases(text_label):
    keyphrases = {}
    for i, row in text_label.iterrows():
        topic = row['Название документа']
        phrase = row['Фразы']
        skip = row['skip']
        if topic in keyphrases.keys():
            keyphrases[topic].append([phrase, skip])
        else:
            keyphrases[topic] = [[phrase, skip]]
    return list(keyphrases.items())


def find_topic_and_keyphrases(text, keyphrases):
    predict_topic = []
    func_sorting = lambda a: np.all(a[:-1] <= a[1:])
    for topic, matches in keyphrases:
        for match, skip in matches:
            check_find, position = find_sub_with_skip(text=text,
                                                      key=match,
                                                      skip=skip,
                                                      func_sorting=func_sorting)
            if check_find:
                predict_topic.append([topic, position[1]])
    if len(predict_topic) != 0:
        # unique_predict_topics = list(dict.fromkeys([i for i in predict_topic]))
        # return unique_predict_topics
        return predict_topic
    return ['Невозможно определить']


def get_head_contour(pil_image):
    image = np.array(pil_image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(gray, 200, 255, 0)
    kernel = np.ones((20, 20), 'uint8')
    erode_img = cv2.erode(gray, kernel, cv2.BORDER_REFLECT, iterations=2)
    contours = cv2.findContours(erode_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    # cv2.drawContours(image, contours, -1, (0,255,0), 3)
    # cv2.drawContours(erode_img, contours, -1, (0,255,0), 3)
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
    image = cv2.resize(image, (700, 1000))
    cv2.imshow('image', image)
    # cv2.imshow('image', image)
    cv2.waitKey()


if __name__ == '__main__':
    # PDF_DOC = 'example/20220616142153936.pdf'
    PDF_DOC = 'example/20220623133012883.pdf'
    # PDF_DOC = 'example/20220621152729476.pdf'
    # PDF_DOC = 'example/20220616150206237.pdf'
    PATH_POPLER = 'poppler-22.04.0/Library/bin'
    # pdf_dor = 'C:/Users/Asilidae/Desktop/bank/example/20220616142153936.pdf'
    PATH_ENGINE = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
    PATH_PHRASES = 'phrases.xlsx'

    phrases_topic = pd.read_excel(PATH_PHRASES)
    keyphrases_topic = get_keyphrases(phrases_topic)

    tool, lang = init_ocr(PATH_ENGINE)
    images = convert_from_path(PDF_DOC, poppler_path=PATH_POPLER)
    text = get_text_from_images(images, lang, pyocr.builders.TextBuilder())
    prep_text = prepare_text(text)
    # print(prep_text)

    result = find_topic_and_keyphrases(prep_text, keyphrases_topic)
    sorted_ids = sorted(result, key=lambda x: x[1])[0]
    print(sorted_ids)
