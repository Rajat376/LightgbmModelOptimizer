class StringFileReader:
    def __init__(self, string_data):
        self.lines = string_data.splitlines()
        self.index = 0

    def readline(self)->str:
        if self.index < len(self.lines):
            current_line = self.lines[self.index]
            self.index += 1
            return current_line + '\n'  # Add newline character to match file.readline() behavior
        else:
            return ''
    def reset(self):
        self.index = 0


class StringFileWriter:
    def __init__(self):
        self.lines = []

    def write(self, line):
        self.lines.append(line)

    def writeline(self,text):
        self.lines.append(text)
        self.lines.append("\n")

    def __str__(self):
        return ''.join(self.lines)