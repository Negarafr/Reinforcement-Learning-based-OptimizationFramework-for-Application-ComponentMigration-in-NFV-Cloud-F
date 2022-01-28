
import numpy as np
from environment import *


class EnvironmentChanger(object):

    def __init__(self, request, minServiceLength, maxServiceLength, num_vnfs):
        self.request = request
        self.minServiceLength = minServiceLength
        self.maxServiceLength = maxServiceLength
        self.num_vnfs = num_vnfs
        self.serviceLength = np.zeros(self.request,  dtype='int32')
        self.state = np.zeros((self.request, self.maxServiceLength),  dtype='int32')

    def getNewState(self):
        self.serviceLength = np.zeros(self.request,  dtype='int32')
        self.state = np.zeros((self.request, self.maxServiceLength),  dtype='int32')
        for batch in range(self.request):
            self.serviceLength[batch] = np.random.randint(self.minServiceLength, self.maxServiceLength+1, dtype='int32')
            VnfID=[0,1,2,3,4,5,6]
            for i in range(self.serviceLength[batch]):
                self.state[batch][i] = VnfID[i]

if __name__ == "__main__":

    # Define generator
    request = 1
    minServiceLength = 1
    maxServiceLength = 4
    num_vnfds = 3
    env = EnvironmentChanger(request, minServiceLength, maxServiceLength, num_vnfds)
    env.getNewState()

import numpy as np
from environment import *
#ServiceBatchGenerator

class EnvironmentChanger(object):

    def __init__(self, request, minServiceLength, maxServiceLength, num_vnfs):
        self.request = request
        self.minServiceLength = minServiceLength
        self.maxServiceLength = maxServiceLength
        self.num_vnfs = num_vnfs
        self.serviceLength = np.zeros(self.request,  dtype='int32')
        self.state = np.zeros((self.request, self.maxServiceLength),  dtype='int32')

    def getNewState(self):
        self.serviceLength = np.zeros(self.request,  dtype='int32')
        self.state = np.zeros((self.request, self.maxServiceLength),  dtype='int32')
        for batch in range(self.request):
            self.serviceLength[batch] = np.random.randint(self.minServiceLength, self.maxServiceLength+1, dtype='int32')
            VnfID=[0,1,2,3,4,5,6]
            for i in range(self.serviceLength[batch]):
                self.state[batch][i] = VnfID[i]

if __name__ == "__main__":

    # Define generator
    request = 1
    minServiceLength = 1
    maxServiceLength = 4
    num_vnfds = 3
    env = EnvironmentChanger(request, minServiceLength, maxServiceLength, num_vnfds)
    env.getNewState()
def get_all_config_parameters():{
    #customize this function based on the inputs parameters which change in each time-slot
    pass
    }



