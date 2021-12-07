class Counter:
    def __init__(self, xCoord = 0, yCoord = 0, xSize = 0, ySize = 0):
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.peopleInsideTheBuilding = 0
        self.xCoord = xCoord
        self.yCoord = yCoord
        self.xSize = xSize
        self.ySize = ySize

    def checkCollision(self,x, y):
        if x>self.xCoord and x<(self.xSize + self.xCoord) and y>self.yCoord and y<self.yCoord + self.ySize:
           return True

        return False


    def EnterBuilding(self, x, y, prx, pry):
        if not self.checkCollision(x, y):
            if self.checkCollision(prx, pry):
                self.peopleInsideTheBuilding += 1

    def ExitBuilding(self, x, y, prx, pry):
        if self.checkCollision(x, y):
            if not self.checkCollision(prx, pry):
                self.peopleInsideTheBuilding -= 1

    def RegisterAction(self, x, y, prx, pry):
       self.EnterBuilding(x, y, prx, pry)
       self.ExitBuilding(x, y, prx, pry)








