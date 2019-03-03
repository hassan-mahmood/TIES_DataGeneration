import imgkit


options = {
'crop-h': 2000,
'crop-w': 2000,
'crop-x': 2,
'crop-y': 2,
}
imgkit.from_file('myfile.html', 'output.jpg', options=options)