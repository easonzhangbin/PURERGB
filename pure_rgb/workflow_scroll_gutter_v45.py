"""Reserve real sidebar space between scrolling controls and the scrollbar."""

from PyQt6.QtWidgets import QScrollArea


def reserve_scrollbar_gutter(scroll, gutter=12):
    """Keep at least ``gutter`` logical points beside the content viewport.

    This shrinks only a sidebar viewport. Widget-resizable pages retain their
    natural layout and scroll coordinates; preview/canvas coordinates are not
    involved. Repeated calls preserve any larger existing reservation.
    """
    if not isinstance(scroll, QScrollArea):
        raise TypeError("Sidebar gutter requires a QScrollArea")
    if type(gutter) is not int or not 0 <= gutter <= 64:
        raise ValueError("Sidebar scrollbar gutter must be an integer from 0 to 64")
    margins = scroll.viewportMargins()
    right = max(margins.right(), gutter)
    scroll.setViewportMargins(margins.left(), margins.top(), right, margins.bottom())
    return scroll
