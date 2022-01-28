import numpy as np
import pickle as pic
import math

# location Fog Nodes
infile = open('Gauss', 'rb')
Location_Fog = pic.load(infile)
infile.close()
# Location Cloud Nodes
infile = open('Location_cloud', 'rb')
location_Cloud = pic.load(infile)
infile.close()
# location IoT
infile = open('Random_way', 'rb')
Location_IoT = pic.load(infile)
infile.close()

def distance(simulationtime, IoTN, point1):
    x1 = Location_IoT[IoTN][simulationtime][0]
    y1 = Location_IoT[IoTN][simulationtime][1]
    z1 = Location_IoT[IoTN][simulationtime][2]
    x2 = point1[0]
    y2 = point1[1]
    z2 = point1[2]

    x = abs(x1) - abs(x2)
    y = abs(y1) - abs(y2)
    z = abs(z1) - abs(z2)
    dis = math.sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2))
    return round(dis,2)
def distanceIpvnF(point1,point2):
    x1 = point1[0]
    y1 = point1[1]
    z1 = point1[2]
    x2 = point2[0]
    y2 = point2[1]
    z2 = point2[2]
    x = abs(x1) - abs(x2)
    y = abs(y1) - abs(y2)
    z = abs(z1) - abs(z2)
    dis = math.sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2))
    return round(dis,2)


class Environment(object):

    def __init__(self, num_nodes, num_vnfds, num_IoT, num_LinkEdge, num_LinkIoT,avail_nodeCap_Memory,avail_nodeCap_Core,dv_prev_array):

        # Environment properties
        self.num_nodes = num_nodes
        self.num_vnfds = num_vnfds
        self.num_IoT = num_IoT
        self.num_LinkEdge = num_LinkEdge
        self.num_LinkIoT = num_LinkIoT
        #self.node_properties = [{"NodeID": 0, "node_type": ' ', "node_loc": [], "node_procDelay": 0,"nodeCap_Core": 0, "nodeCap_Memory": 0,'Pmax':0,'pidel':0} for _ in range(num_nodes)]
        self.IoT_properties= [{"Location":0 } for _ in range(num_LinkIoT)]
        self.IoTLink_properties = [{"LinkID": 0, "Type": '', "BW": 0, 'edgeside1': 0, 'edgeside2': 0} for _ in range(num_LinkIoT)]
        #self.nodeLink_properties = [{"LinkIDNode": 0, "Type": '', "BW": 0, 'edgeside1': 0, 'edgeside2': 0} for _ in range(num_LinkEdge)]
        self.vnfd_properties = [  {"VNFID": 0, "memory": 0, "resource": 0, "ipvnf": 0, "IoTconnection": [],"trafficIoT": 0, "trafficIpvn": 0, "traffcTotal": 0} for _ in range(num_vnfds)]
        self.avail_nodeCap_Memory = avail_nodeCap_Memory
        self.avail_nodeCap_Core = avail_nodeCap_Core
        self.dv_prev_array=dv_prev_array
        self.assignedNodeVNF = np.zeros([num_vnfds])
        self.len_a_timeslot = len_a_timeslot
        self.BW_ocuupation_Cost = BW_ocuupation_Cost
        self.Energy_Consum_Cost = Energy_Consum_Cost
        self.temp1 = 0
        self.VNFIoTConnection = np.zeros((num_vnfds, num_IoT), dtype=bool) #VNF and IoT connection,If True there is a connection
        self.VNFIoTConnection[1, 0] = True
        self.delayprocessing = 0
        self.transmissionDelay=0
        self.migrationdelay=0
        self.propagDelayIoT =0
        self.alfaobject= 0.9      # Assign environmental properties
        self.topology_json=self.create_json_topology() #YAFS
        self._getNodeProperties(num_nodes)
        self._getVnfdProperties(num_vnfds)
        self.getTraffic()
        self.get_link(num_nodes)
        self.get_IoTlink(num_nodes, num_IoT)
        self.invalidPlacement=False


        # Environment cell slots
        self.max_slots = max([node["nodeCap_Memory"] for node in self.topology_json["entity"]])
        self.cells = np.empty((self.num_nodes, self.max_slots))
        self.cells[:] = np.nan
    def create_json_topology(self):
        ## MANDATORY FIELDS
        topology_json = {}
        topology_json["entity"] = []
        topology_json["link"] = []
        topology_json["entity"] = [
            {"NodeID": 0, "model": ' ', "node_loc": [], "node_procDelay": 0, "nodeCap_Core": 0, "nodeCap_Memory": 0,
             'Pmax': 0, 'pidel': 0} for _ in range(num_nodes)]
        topology_json["link"] = [{"s": 0, "d": 0, "BW": 0, "PR": 0, 'Type': ' '} for _ in range(num_LinkEdge)]

        return topology_json
    def _getNodeProperties(self,num_nodes):
        for i in range(num_nodes):
            self.topology_json["entity"][i]["NodeID"] = i
            self.topology_json["entity"][i]["model"] = node_Type[i]
            self.topology_json["entity"][i]["node_loc"] = node_loc
            self.topology_json["entity"][i]["node_procDelay"] = node_procDelay[i]
            self.topology_json["entity"][i]["nodeCap_Core"] = nodeCap_Core[i]
            self.topology_json["entity"][i]["nodeCap_Memory"] = nodeCap_Memory[i]
            self.topology_json["entity"][i]["Pmax"] = nodePmax[i]
            self.topology_json["entity"][i]["pidel"] = nodepidel[i]

        cloudNum = 0
        FogNum = 0
        for i in range(num_nodes):
            # FogNumber = 2
            # CloudNumber = 2
            if self.topology_json["entity"][i]["model"] == 'Cloud':
                self.topology_json["entity"][i]["node_loc"] = location_Cloud[cloudNum]
                cloudNum = cloudNum + 1
            else:
                self.topology_json["entity"][i]["node_loc"] = Location_Fog[:][FogNum]   # Location_Fog[simulationtime][nodeN][0]
                FogNum += 1

    def get_link(self,num_nodes):
        ed = 0
        for e1 in range(num_nodes - 1):
            for e2 in range(e1 + 1, num_nodes):
                if self.topology_json["entity"][e1]["model"] == 'Fog' and self.topology_json["entity"][e2]["model"] == 'Fog':
                    self.topology_json["link"][ed]['Type'] = 'Fog-Fog'
                    self.topology_json["link"][ed]['s'] = e1
                    self.topology_json["link"][ed]['d'] = e2
                    self.topology_json["link"][ed]['BW'] = 1
                    self.topology_json["link"][ed]['PR'] = 1
                    ed += 1
                elif self.topology_json["entity"][e1]["model"] == 'Fog' and self.topology_json["entity"][e2]["model"] == 'Cloud':
                    self.topology_json["link"][ed]['Type'] = 'Fog-Cloud'
                    self.topology_json["link"][ed]['s'] = e1
                    self.topology_json["link"][ed]['d'] = e2
                    self.topology_json["link"][ed]['BW'] = 5
                    self.topology_json["link"][ed]['PR'] = 1
                    ed += 1
                else:
                    self.topology_json["link"][ed]['Type'] = 'Cloud-Cloud'
                    self.topology_json["link"][ed]['s'] = e1
                    self.topology_json["link"][ed]['d'] = e2
                    self.topology_json["link"][ed]['BW'] = 10
                    self.topology_json["link"][ed]['PR'] = 3
                    ed += 1

    def _getVnfdProperties(self, num_vnfds):
        for i in range(num_vnfds):
            self.vnfd_properties[i]["VNFID"] = i
            self.vnfd_properties[i]["memory"] = vnfmemory[i]
            self.vnfd_properties[i]["resource"] = vnfresource[i]
            self.vnfd_properties[i]["ipvnf"] = IPVNFLIST[i]
            self.vnfd_properties[i]["IoTconnection"] = VNFIoTConnection[i]
    def get_IoTlink(self, num_nodes, num_IoT):
        edIoT = 0
        for e1 in range(num_IoT):
            for e2 in range(num_nodes):
                if self.topology_json["entity"][e2]["model"] == 'Fog':
                    self.IoTLink_properties[edIoT]['LinkID'] = edIoT
                    self.IoTLink_properties[edIoT]['edgeside1'] = e1
                    self.IoTLink_properties[edIoT]['edgeside2'] = e2
                    self.IoTLink_properties[edIoT]['BW'] = 1
                    self.IoTLink_properties[edIoT]['Type'] = "IoT-Fog"
                    edIoT += 1

                elif self.topology_json["entity"][e2]["model"] == 'Cloud':
                    self.IoTLink_properties[edIoT]['LinkID'] = edIoT
                    self.IoTLink_properties[edIoT]['edgeside1'] = e1
                    self.IoTLink_properties[edIoT]['edgeside2'] = e2
                    self.IoTLink_properties[edIoT]['BW'] = 10
                    self.IoTLink_properties[edIoT]['Type'] = "IoT-Cloud"
                    edIoT += 1
    def getTraffic(self):
        #IOT Traffic
        for v in range(self.num_vnfds):
            for iot in range(self.num_IoT):
                if self.vnfd_properties[v]["IoTconnection"][iot] == True:
                    self.vnfd_properties[v]['trafficIoT'] = self.vnfd_properties[v]['trafficIoT'] + TrafficIoT
                else:
                    self.vnfd_properties[v]['trafficIoT'] = self.vnfd_properties[v]['trafficIoT'] + 0
        #IPVNF Traffic
        for v in range(self.num_vnfds):
            if v == 0: #first VNF (TrafficIPVNF=0)
                self.vnfd_properties[v]['trafficIpvn'] = 0
                self.vnfd_properties[v]['traffcTotal'] = self.vnfd_properties[v]['trafficIoT']
            elif v > 0:
                for ipvn in range(len(self.vnfd_properties[v]["ipvnf"])):
                    self.temp1 = self.vnfd_properties[v]["ipvnf"].__getitem__(ipvn)
                    self.vnfd_properties[v]['trafficIpvn'] = self.vnfd_properties[v]['trafficIpvn'] + \
                                                             self.vnfd_properties[self.temp1]['traffcTotal']
                    self.vnfd_properties[v]['traffcTotal'] = self.vnfd_properties[v]['trafficIoT'] + \
                                                             self.vnfd_properties[v]['trafficIpvn']
    def getNodeVNF(self):
        for vnf in range(self.num_vnfds):
            for node in range(self.num_nodes):
                if (self.dv_array[node, vnf] == 1):
                    self.assignedNodeVNF[vnf] = node
                    break
        return self.assignedNodeVNF
    def _placeVNF(self, i, node, vnf):
        """ Place Packet """
        for slot in range(self.vnfd_properties[vnf]["memory"]):
            occupied_slot = None
            for slot in range(self.topology_json["entity"][node]["nodeCap_Memory"]):
                if np.isnan(self.cells[node][slot]):
                    self.cells[node][slot] = vnf
                    occupied_slot = slot
                    break
                elif slot == len(self.cells[node]) - 1:
                    self.invalidPlacement = True
                    occupied_slot = -1  # No space available
                    break
                else:
                    pass
            # Anotate first slot used by the Packet
            if slot == 0:
                self.first_slots[i] = occupied_slot
    def getDelayprocess(self, vnf):
        for node in range(self.num_nodes):
            if self.dv_array[node, vnf] == 1:
                self.delayprocessing = self.vnfd_properties[vnf]['traffcTotal'] * self.topology_json["entity"][node][
                    "node_procDelay"] * self.dv_array[node, vnf]
        return self.delayprocessing
    def getTransmisionDelay(self, vnf,time):  # for iot in range(num_IoT):
        transDelayIoT =0
        propag=[None] * self.num_IoT
        TransDelayIoTTotall=[None]*self.num_IoT
        self.transmissionDelay=0
        '''Transmision Delay IoT'''
        for c in range(len(self.topology_json["entity"])):
            for iot in range(self.num_IoT):
                if self.dv_array[c, vnf] == 1:
                   
                    # Transmission Delay IoT
                    if self.vnfd_properties[vnf]['IoTconnection'][iot] == True:
                        if self.topology_json["entity"][c]["model"] == 'Fog':
                            dis = distance(time, iot, self.topology_json["entity"][c]["node_loc"][time])
                        elif self.topology_json["entity"][c]["model"] == 'Cloud':
                            dis = distance(time, iot, self.topology_json["entity"][c]["node_loc"])
                        self.propagDelayIoT = dis / speed_propag
                        propag[iot]=dis / speed_propag
                        for i in range(len(self.IoTLink_properties)):
                            if self.IoTLink_properties[i]['edgeside1'] == iot and self.IoTLink_properties[i][
                                'edgeside2'] == c:
                                transDelayIoT = transDelayIoT + self.dv_array[c, vnf] * (self.vnfd_properties[vnf]['traffcTotal'] / (self.IoTLink_properties[i]['BW'] - 0))
                        TransDelayIoTTotall[iot]=transDelayIoT+propag[iot]
                    else:
                        TransDelayIoTTotall[iot] = 0
        self.transDelayIoT = max(TransDelayIoTTotall)
        '''Transmission Delay IpVNF'''
        transDelayIpvnf=[None]*len(self.vnfd_properties[vnf]['ipvnf'])
        propagDelayIpvnf=[None]*len(self.vnfd_properties[vnf]['ipvnf'])
        for c in range(len(self.topology_json["entity"])):
            if self.dv_array[c, vnf] == 1:
                if self.vnfd_properties[vnf]['ipvnf'] == []:
                    self.transDelayIpvnf=0
                    #transDelayIpvnf[i1]=0
                    self.transmissionDelay = max(self.transDelayIoT, self.transDelayIpvnf)
                else: #Ipvnf exist
                    #Propagation Delay Ipvnf
                    for i1 in range(len(self.vnfd_properties[vnf]['ipvnf'])):
                        tem=self.getNodeVNF()
                        ipvnf=int(tem[self.vnfd_properties[vnf]['ipvnf'][i1]])
                        if c==ipvnf: #vnf anf IpVnF are in the same node
                            propagDelayIpvnf[i1]=0
                            self.transDelayIpvnf=0
                        else:#vnf anf IpVnF are not in the same node
                                if(self.topology_json["entity"][c]["model"]=='Fog' and self.topology_json["entity"][ipvnf]["model"]=='Cloud'):
 
                                        disip=distanceIpvnF(self.topology_json["entity"][c]["node_loc"][time],self.topology_json["entity"][ipvnf]["node_loc"])
                                elif(self.topology_json["entity"][c]["model"]=='Cloud' and self.topology_json["entity"][ipvnf]["model"]=='Fog'):
                                        disip=distanceIpvnF(self.topology_json["entity"][c]["node_loc"],self.topology_json["entity"][ipvnf]["node_loc"][time])
                                elif(self.topology_json["entity"][c]["model"]=='Cloud' and self.topology_json["entity"][ipvnf]["model"]=='Cloud'):
                                         disip=distanceIpvnF(self.topology_json["entity"][c]["node_loc"],self.topology_json["entity"][ipvnf]["node_loc"])
                                else:
                                    disip=distanceIpvnF(self.topology_json["entity"][c]["node_loc"][time],self.topology_json["entity"][ipvnf]["node_loc"][time])

                                propagDelayIpvnf[i1]=disip/speed_propag
                    propagIp = max(propagDelayIpvnf)


                    #communication Delay IpVNf
                    for i in range(len(self.topology_json["link"])):
                        for i1 in range(len(self.vnfd_properties[vnf]['ipvnf'])):
                            tem = self.getNodeVNF()
                            ipvnf = int(tem[self.vnfd_properties[vnf]['ipvnf'][i1]])
                            if c == ipvnf:
                                transDelayIpvnf[i1]=0
                            else:
                                if self.topology_json["link"][i]['s'] == c and self.topology_json["link"][i]['d'] ==ipvnf or self.topology_json["link"][i]['d'] == c and self.topology_json["link"][i]['s'] ==ipvnf:
                                    transDelayIpvnf[i1]=(self.vnfd_properties[vnf]['trafficIpvn'] / self.topology_json["link"][i]['BW']) * self.dv_array[c, vnf]

                    transIpVnf=max(transDelayIpvnf)


                    self.transDelayIpvnf=transIpVnf+propagIp
                    self.transmissionDelay=max(self.transDelayIoT,self.transDelayIpvnf)

        return self.transmissionDelay
    def getPropagIpvnf(self, vnf, time):
        propagDelayIpvnf = [None] * len(self.vnfd_properties[vnf]['ipvnf'])
        for c in range(len(self.topology_json["entity"])):
            if self.dv_array[c, vnf] == 1:
                if self.vnfd_properties[vnf]['ipvnf'] == []:
                    self.propagIpVNF = 0
                else:  # Ipvnf exist
                    for i1 in range(len(self.vnfd_properties[vnf]['ipvnf'])):
                        tem = self.getNodeVNF()
                        ipvnf = int(tem[self.vnfd_properties[vnf]['ipvnf'][i1]])
                        if c == ipvnf:  # vnf anf IpVnF are in the same node
                            propagDelayIpvnf[i1] = 0
                        else:  # vnf anf IpVnF are not in the same node
                            if (self.topology_json["entity"][c]["model"] == 'Fog' and self.topology_json["entity"][ipvnf][
                                "model"] == 'Cloud'):
                                disip = distanceIpvnF(self.topology_json["entity"][c]["node_loc"][time],
                                                      self.topology_json["entity"][ipvnf]["node_loc"])
                            elif (self.topology_json["entity"][c]["model"] == 'Cloud' and self.topology_json["entity"][ipvnf][
                                "model"] == 'Fog'):
                                disip = distanceIpvnF(self.topology_json["entity"][c]["node_loc"],
                                                      self.topology_json["entity"][ipvnf]["node_loc"][time])
                            elif (self.topology_json["entity"][c]["model"] == 'Cloud' and self.topology_json["entity"][ipvnf][
                                "model"] == 'Cloud'):
                                disip = distanceIpvnF(self.topology_json["entity"][c]["node_loc"],
                                                      self.topology_json["entity"][ipvnf]["node_loc"])
                            else:
                                disip = distanceIpvnF(self.topology_json["entity"][c]["node_loc"][time],
                                                      self.topology_json["entity"][ipvnf]["node_loc"][time])

                            propagDelayIpvnf[i1] = disip / speed_propag


                    self.propagIpVNF = max(propagDelayIpvnf)
        return self.propagIpVNF
    def getMigrationDelay(self, vnf, time):
        for node in range(self.num_nodes):
            if np.array_equal(self.dv_array[node, vnf], self.dv_prev_array[node, vnf]) and self.dv_array[
                node, vnf] == 1:  #: 1 in self.dv_array[c, vnf] :
                self.migrationdelay = 0
            elif self.dv_array[node, vnf] == 1:
                dproc = self.getDelayprocess(vnf)
                dcom = self.getTransmisionDelay(vnf, time)

                node_host_pre = np.where(self.dv_prev_array[:, vnf] == 1)  # node host previous

                node_host_previous = node_host_pre[0][0]

                if dproc + dcom <= self.len_a_timeslot:
                    self.migrationdelay = 0
                    break

                elif dcom >= self.len_a_timeslot:
                    ''''Migration old Traffic'''''
                    newtraffic = self.vnfd_properties[vnf]['traffcTotal']
                    for i in range(self.num_LinkEdge):
                        if self.topology_json["link"][i]['s'] == node and self.topology_json["link"][i]['d'] == \
                                node_host_previous or self.topology_json["link"][i]['d'] == node and \
                                self.topology_json["link"][i]['s'] == node_host_previous:


                            availabebw = self.topology_json["link"][i]['BW']
                            procedelayold = newtraffic * self.topology_json["entity"][node]["node_procDelay"] * self.dv_prev_array[
                                node_host_previous, vnf]
                    procdelaynew = newtraffic * self.topology_json["entity"][node]["node_procDelay"] * self.dv_array[node, vnf]
                    self.migrationdelay = (newtraffic / availabebw) + procdelaynew - procedelayold
                    self.migrationdelay = self.getPropagIpvnf(vnf, time) + (
                                self.vnfd_properties[vnf]['memory'] / availabebw) + (newtraffic * procdelaynew) - (
                                                      procdelaynew * procedelayold)
                    break

                else:
                    '''Migration new Traffic'''
                    dproc = self.getDelayprocess(vnf)
                    dcom = self.getTransmisionDelay(vnf, time)
                    a = (self.len_a_timeslot - dcom) / (dproc)
                    b = max(0,
                            self.vnfd_properties[vnf]['traffcTotal'] - (a * self.vnfd_properties[vnf]['traffcTotal']))
                    newtraffic = min(self.vnfd_properties[vnf]['traffcTotal'], b)

                    for i in range(len(self.topology_json["link"])):
                        if self.topology_json["link"][i]['edgeside1'] == node and self.topology_json["link"][i]['edgeside2'] == \
                                node_host_previous or self.topology_json["link"][i]['edgeside2'] == node and \
                                self.topology_json["link"][i]['edgeside1'] == node_host_previous:

                            availabebw = self.topology_json["link"][i]['BW']

                    procedelayopld = newtraffic * self.topology_json["entity"][node]["node_procDelay"] * self.dv_prev_array[
                        node_host_previous, vnf]
                    procdelaynew = newtraffic * self.topology_json["entity"][node]["node_procDelay"] * self.dv_array[node, vnf]


                    self.migrationdelay = self.getPropagIpvnf(vnf, time) + (
                            self.vnfd_properties[vnf]['memory'] / availabebw) + (newtraffic * procdelaynew) - (procdelaynew * procedelayopld)
                    break

        return self.migrationdelay
    def getMigrationCost(self, vnf):
        for node in range(self.num_nodes):
            if self.dv_array[node,vnf] == self.dv_prev_array[node, vnf] and self.dv_array[node, vnf] == 1:
                self.migrationCost = 0
            elif self.dv_array[node, vnf] == 1:
                self.migrationCost = self.vnfd_properties[vnf]['memory'] * self.BW_ocuupation_Cost

        return self.migrationCost
    def getPowerConsumptionCost(self, vnf):
        for node in range(self.num_nodes):
            if self.dv_array[node, vnf] == 1:
                self.resource_utilization = min(1, self.vnfd_properties[vnf]['resource'] / self.topology_json["entity"][node]['nodeCap_Core'])
                self.powercost = self.Energy_Consum_Cost * self.topology_json["entity"][node]["pidel"]  + (self.topology_json["entity"][node]["Pmax"]  - self.topology_json["entity"][node]["pidel"] ) * self.resource_utilization
        return self.powercost
    def _computeReward(self,time):
        #sequence
        '''Reward'''
        DelayProcessRoot=0
        DelayTransRoot=0
        DelayMigrationRoot=0
        CostMigrationRoot = 0
        CostPowerRoot = 0
        w1D=0
        w2D=0
        w3D=0
        w4C=0
        w5C=0

        # seq
        for v in range(len(self.vnfd_properties)):
            '''Processing Delay'''
            w1D=w1D+self.getDelayprocess(v)
            '''Transmission Delay'''
            w2D=w2D+abs(((self.getTransmisionDelay(v,time)-XminDT)/(XmaxDT-XminDT)))
            '''Power Consumption Cost'''
            w5C = w5C + abs(((self.getPowerConsumptionCost(v) - XminCP) / (XmaxCP - XminCP)))
            '''Migration Delay'''
            if self.getMigrationDelay(v,time)==0:
                w3D = w3D+0
                #'Delayprocess Weight',self.getDelayprocess(v), 'weightTransDelay',(self.getTransmisionDelay(v, time) - XminDT) / (XmaxDT - XminDT), 'weightMigrationDelay',0)
            else:
                w3D = w3D + abs(((self.getMigrationDelay(v, time) - XminDM) / (XmaxDM - XminDM)))
                #'Delayprocess Weight', self.getDelayprocess(v), 'weightTransDelay',(self.getTransmisionDelay(v, time) - XminDT) / (XmaxDT - XminDT), 'weightMigrationDelay',(self.getMigrationDelay(v, time) - XminDM) / (XmaxDM - XminDM))

            '''Migration Cost'''
            if self.getMigrationCost(v)==0:

                w4C = w4C+0
                #'Weight MigrationCost', 0, 'weight CostPower', self.getPowerConsumptionCost(v)
            else:

               # 'Weight MigrationCost', abs(((self.getMigrationCost(v)-xminCM)/(xmaxCM-xminCM))), 'weight CostPower',self.getPowerConsumptionCost(v))

                w4C=w4C+abs(((self.getMigrationCost(v)-xminCM)/(xmaxCM-xminCM)))

            #DelayProcessRoot=DelayProcessRoot+self.getDelayprocess(v)
            #DelayTransRoot=DelayTransRoot+self.getTransmisionDelay(v,time)
            #DelayMigrationRoot = DelayMigrationRoot + self.getMigrationDelay(v,time)
            #CostMigrationRoot = CostMigrationRoot + self.getMigrationCost(v)
            #CostPowerRoot = CostPowerRoot + self.getPowerConsumptionCost(v)

        '''*____Root Cal*____'''
        WDtotal = w1D + w2D + w3D
        WCtotal = w4C + w5C
        self.DelayRequest=WDtotal/(self.chain_length)
        self.CostRequest=WCtotal/(self.chain_length)
        self.objec=(self.alfaobject*self.DelayRequest)+((1-self.alfaobject)*self.CostRequest)
        return self.objec,self.DelayRequest,self.CostRequest,(w5C/self.chain_length)
    def step(self, placement, service, chain_length,time):
        """ Place VNF """
        self.chain_length = chain_length
        self.network_service = service
        self.placement = placement
        self.first_slots = -np.ones(chain_length, dtype='int32') #not assigned
        self.dv_array = np.zeros([self.num_nodes, self.num_vnfds])


        for i in range(chain_length):
            #Constraint
            if self.avail_nodeCap_Memory[placement[i]] >= self.vnfd_properties[service[i]]["memory"] and \
                    self.avail_nodeCap_Core[placement[i]] >=self.vnfd_properties[service[i]]["resource"]:
                self._placeVNF(i, placement[i], service[i])
                self.dv_array[placement[i], service[i]] = 1

            else:
                v = service[i]
                n=np.where(self.dv_prev_array[:,v]==1)
                self.dv_array[n[0][0], v] = self.dv_prev_array[n[0][0],v]

        """Compute Assigned VNF"""
        """ Compute reward """
        if self.invalidPlacement == True:
            self.reward = 0
        else:
            self.reward = self._computeReward(time)
    def clear(self):
            # Clear environment
        self.cells = np.empty((self.num_nodes, self.max_slots))
        self.cells[:] = np.nan
        self.objec=0
        self.DelayRequest=0
        self.CostRequest=0

        self.invalidPlacement = False

        # Clear placement
        infile = open('NodeCapMemory', 'rb')
        avail_nodeCap_Memory = pic.load(infile)
        infile.close()
        infile = open('NodeCapCore', 'rb')
        avail_nodeCap_Core = pic.load(infile)
        infile.close()
        self.avail_nodeCap_Memory = avail_nodeCap_Memory
        self.avail_nodeCap_Core = avail_nodeCap_Core

        self.network_service = None
        self.placement = None
        self.first_slots = None
        self.reward = None
        self.invalid_placement = False
        self.link_latency = 0
        self.cpu_latency = 0



if __name__ == "__main__":
    # Define environment
    num_nodes = 3
    num_vnfds = 3
    num_IoT = 1
    num_LinkEdge = (num_nodes * (num_nodes - 1)) // 2
    num_LinkIoT = num_nodes * num_IoT
    vnfavamemory =[1, 2,2,1,2,1,2]   # GB
    vnfresource =[3, 2,3,1,3,3,3]  # VNF resource requirement
    avail_nodeCap_Core = [4, 4,4, 4,4, 4,4, 4,4, 4, 8, 8, 8]
    avail_nodeCap_Memory =[4, 4,4, 4,4, 4,4, 4,4, 4, 8, 8, 8]

    dv_prev_array = np.zeros([num_nodes, num_vnfds])
    dv_prev_array[2, 0] = 1
    dv_prev_array[0, 1] = 1
    dv_prev_array[0, 2] = 1
    for n in range(num_nodes):
        for v in range(num_vnfds):
            if dv_prev_array[n, v] == 1:
                avail_nodeCap_Memory[n] = avail_nodeCap_Memory[n] - vnfavamemory[v]
                avail_nodeCap_Core[n] = avail_nodeCap_Core[n] - vnfresource[v]
    env = Environment(num_nodes, num_vnfds, num_IoT, num_LinkEdge, num_LinkIoT,avail_nodeCap_Memory,avail_nodeCap_Core,dv_prev_array)

    # Allocate service in the environment
    service_length = 3
    network_service = [0, 1, 2]
    placement = [0, 1, 2]
    env.step(placement,network_service,service_length,0)
    env.clear()


