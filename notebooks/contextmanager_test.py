from contextlib import AbstractContextManager

#code within a "with" block is run in the scope of the enter.

class FileManager(AbstractContextManager):
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None

    def __enter__(self):
        print(f"Opening file: {self.filename}")
        self.file = open(self.filename, self.mode)
        return self.file
    
    def __exit__(self, exc_type, exc_value, traceback):
        #if file has been created in the __enter__ process
        if self.file:
            print(f"Closing file: {self.filename}")
            self.file.close()
        #raises an exception
        return False
    
filename = "example.txt"

with FileManager(filename, 'w') as file:
    file.write("hello world!")
