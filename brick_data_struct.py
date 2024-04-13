class Brick():
    def __init__(id, init_string = None):
        this.id = id
        if init_string != None:
            index = init_string.find(' ')
            if index != -1: 
                first_part = init_string[:index]
                second_part = init_string[index+1:]
                this.part_number = first_part
                this.name = second_part
                return
        this.part_number = None
        this.name = None
