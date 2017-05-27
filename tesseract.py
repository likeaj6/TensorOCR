from PIL import Image
import sys

import pyocr
import pyocr.builders

tools = pyocr.get_available_tools()
if len(tools) == 0:
    print("No OCR tool found")
    sys.exit(1)
# The tools are returned in the recommended order of usage
tool = tools[0]
print("Will use tool '%s'" % (tool.get_name()))
# Ex: Will use tool 'libtesseract'

langs = tool.get_available_languages()
print("Available languages: %s" % ", ".join(langs))
lang = langs[0]
print("Will use lang '%s'" % (lang))
# Ex: Will use lang 'fra'
# Note that languages are NOT sorted in any way. Please refer
# to the system locale settings for the default language
# to use.


txt = tool.image_to_string(
    Image.open('test0.jpg'),
    lang=lang,
    builder=pyocr.builders.TextBuilder()
)
# txt is a Python string
print("text1" + txt)

txt = tool.image_to_string(
    Image.open('test1.jpg'),
    lang=lang,
    builder=pyocr.builders.TextBuilder()
)
# txt is a Python string
print("text2" + txt)

txt = tool.image_to_string(
    Image.open('test2.jpg'),
    lang=lang,
    builder=pyocr.builders.TextBuilder()
)
# txt is a Python string
print("text3" + txt)
# word_boxes = tool.image_to_string(
#     Image.open('test.png'),
#     lang="eng",
#     builder=pyocr.builders.WordBoxBuilder()
# )
# # list of box objects. For each box object:
# #   box.content is the word in the box
# #   box.position is its position on the page (in pixels)
# #
# # Beware that some OCR tools (Tesseract for instance)
# # may return empty boxes
# print("boxes" + word_boxes)
#
# line_and_word_boxes = tool.image_to_string(
#     Image.open('test.png'), lang="fra",
#     builder=pyocr.builders.LineBoxBuilder()
# )
