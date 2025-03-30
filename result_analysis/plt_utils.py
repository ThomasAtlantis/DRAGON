from matplotlib.font_manager import FontProperties


def Font(face="", size=14, font_path=None):
    # Shell: `fc-list :lang=zh-cn`
    font_pool = {
        'arial': 'fonts/arial.ttf',
        'times new roman': 'fonts/Times.ttc',
    }
    font_path = font_pool.get(face.lower(), font_path)
    return FontProperties(fname=font_path, size=size)
