class Brick():
    def __init__(self, id, init_string = None) -> None:
        self.id = id
        if init_string != None:
            index = init_string.find(' ')
            if index != -1: 
                first_part = init_string[:index]
                second_part = init_string[index+1:]
                self.part_number = first_part
                self.name = second_part
                return
        self.part_number = None
        self.name = None