from screeninfo import get_monitors


def get_screen_size():
    height, width = 0, 0
    for m in get_monitors():
        if m.is_primary:
            height = m.height
            width = m.width
    return height, width