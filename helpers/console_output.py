from PySide6.QtCore import QThread, Signal


class CaptureConsoleOutputThread(QThread):
    """ Captures the contents that would normally be written to the console and emits them as a signal to the GUI. """
    output_written = Signal(str)  # Signal to update GUI

    def __init__(self, process):
        super().__init__()
        self.process = process
        self.running = True

    def run(self):
        """ Starts the capture process """
        buffer = ""
        try:
            while self.running:
                char = self.process.stdout.read(1)
                if char == '' and self.process.poll() is not None:
                    break  # Process finished and nothing left to read

                if char in ('\n', '\r'):
                    if buffer:
                        self.output_written.emit(buffer)
                        buffer = ""
                elif char:
                    buffer += char
        except Exception as e:
            self.output_written.emit(f"Error capturing output: {str(e)}")

    def stop(self):
        """ Stop capturing output """
        self.running = False
        self.quit()
        self.wait()

    def flush(self):
        pass  # Required for compatibility when overriding sys.stdout
