"""Bounded, session-only photo edit commands, separate from saved render jobs."""

from copy import deepcopy


class PhotoEditHistory:
    def __init__(self, snapshot, *, limit=80):
        self.current = deepcopy(snapshot)
        self.commands = []
        self.cursor = 0
        self.limit = limit
        self.depth = 0
        self.group = 0
        self.group_label = None
        self.group_start = None

    def begin(self, label):
        if not self.depth:
            self.group += 1
            self.group_label = label
            # A previewed drag may return to its starting value. Preserve the
            # complete branch so this no-op also keeps redo and a full limit.
            self.group_start = (deepcopy(self.current), deepcopy(self.commands), self.cursor)
        self.depth += 1

    def end(self):
        if not self.depth:
            return
        self.depth = max(0, self.depth - 1)
        if not self.depth:
            self._finish_group()

    def _finish_group(self):
        if self.group_start is not None:
            before, commands, cursor = self.group_start
            if self.current == before:
                self.commands, self.cursor = commands, cursor
            self.group_start = None
        self.depth = 0
        self.group_label = None

    def record(self, snapshot, label):
        after = deepcopy(snapshot)
        if self.current == after:
            return False
        del self.commands[self.cursor:]
        group = self.group if self.depth else None
        if (group is not None and self.cursor
                and self.commands[-1]["group"] == group):
            self.commands[-1]["after"] = after
            if self.commands[-1]["before"] == after:
                self.commands.pop()
                self.cursor -= 1
        else:
            self.commands.append(dict(before=self.current, after=after,
                                      label=self.group_label or label, group=group))
            self.cursor += 1
        if len(self.commands) > self.limit:
            del self.commands[:len(self.commands) - self.limit]
            self.cursor = len(self.commands)
        self.current = deepcopy(after)
        return True

    def absorb(self, snapshot):
        """Keep derived analysis in its user command, rather than a new undo."""
        after = deepcopy(snapshot)
        self.current = after
        if self.cursor:
            self.commands[self.cursor - 1]["after"] = deepcopy(after)
        if self.cursor < len(self.commands):
            self.commands[self.cursor]["before"] = deepcopy(after)

    def can_undo(self):
        return self.cursor > 0

    def can_redo(self):
        return self.cursor < len(self.commands)

    def undo_label(self):
        return self.commands[self.cursor - 1]["label"] if self.can_undo() else ""

    def redo_label(self):
        return self.commands[self.cursor]["label"] if self.can_redo() else ""

    def move(self, redo=False):
        self._finish_group()
        if not (self.can_redo() if redo else self.can_undo()):
            return None
        if redo:
            state = self.commands[self.cursor]["after"]
            self.cursor += 1
        else:
            self.cursor -= 1
            state = self.commands[self.cursor]["before"]
        self.current = deepcopy(state)
        return deepcopy(state)
