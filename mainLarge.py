
import logging
import tensorflow as tf
from environment import *
from Service_batch_generator import *
from agent import *
from config import *

from tensorflow.python import debug as tf_debug
from tqdm import tqdm
import matplotlib.pyplot as plt
from operator import itemgetter


objecttive=[]
objecttiveTotall=[]
objecttiveDelay=[]
objectiveCost=[]
objectiveArrayAll=[]


""" Globals """
DEBUG = True

if __name__ == "__main__":

    """ Log """
    logging.basicConfig(level=logging.DEBUG)
    """ Configuration """
    config, _ = get_config()
    """ Batch of Services """
    services = EnvironmentChanger(config.batch_size, config.min_length, config.max_length, config.num_vnfds)

    """ Agent """
    state_size_sequence = config.max_length
    state_size_embeddings = config.num_vnfds
    action_size = config.num_nodes
    agent = Agent(state_size_embeddings, state_size_sequence, action_size, config.batch_size, config.learning_rate, config.hidden_dim, config.num_stacks)

    """ Configure Saver to save & restore model variables """
    variables_to_save = [v for v in tf.global_variables() if 'Adam' not in v.name]
    saver = tf.train.Saver(var_list=variables_to_save, keep_checkpoint_every_n_hours=1.0)

    print("Starting session...")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters

        # Restore variables from disk
        if config.load_model:
            saver.restore(sess, "save/tf_vnfMigration.ckpt")
            print("Model restored.")

        # Train model
        if config.train_mode:

            # Summary writer
            writer = tf.summary.FileWriter("summary/repo", sess.graph)
            dv_prev_array = np.zeros([config.num_nodes, config.num_vnfds])  # (cpu,vnf)
            #set the first placement Ex: dv_prev_array[6, 0] = 1;
            #Capacity Node
            for n in range(config.num_nodes):
                for v in range(config.num_vnfds):
                    if dv_prev_array[n, v] == 1:
                        avail_nodeCap_Memory[n] = avail_nodeCap_Memory[n] - vnfavamemory[v]
                        avail_nodeCap_Core[n] = avail_nodeCap_Core[n] - vnfresource[v]

            outfile = open('NodeCapMemory', 'wb')
            pic.dump(avail_nodeCap_Memory, outfile)
            outfile.close()

            outfile = open('NodeCapCore', 'wb')
            pic.dump(avail_nodeCap_Core, outfile)
            outfile.close()

            # Main Loop
            print("\n Starting training...")
            for time in range(config.time_step):
                rewardarray = []
                rewardArrayAll=[]
                delayarray=[]
                costarray=[]
                positionarray = []
                pre_placement = np.zeros([config.num_nodes, config.num_vnfds])

                env = Environment(config.num_nodes, config.num_vnfds, config.num_IoT, config.num_LinkEdge,
                                  config.num_LinkIoT, avail_nodeCap_Memory,avail_nodeCap_Core,dv_prev_array)

                for e in tqdm(range(config.num_epoch)):
                    # New batch of states
                    services.getNewState()
                    # Vector embedding
                    input_state = vector_embedding(services)
                    # Compute placement
                    feed = {agent.input_: input_state, agent.input_len_: [item for item in services.serviceLength]}
                    positions = sess.run(agent.ptr.positions, feed_dict=feed)
                    positionarray.append(positions)
                    reward = np.zeros(config.batch_size)

                    # Compute environment
                    for batch in range(config.batch_size):
                        env.clear()
                        env.step(positions[batch], services.state[batch], 10, time)#2=chain lenght
                        rewardArrayAll.append(env.reward)
                    # RL Learning
                    feed = {agent.reward_holder: [item for item in reward], agent.positions_holder: positions,
                            agent.input_: input_state, agent.input_len_: [item for item in services.serviceLength]}
                    summary, _ = sess.run([agent.merged, agent.train_step], feed_dict=feed)

                    if e % 100 == 0:
                        writer.add_summary(summary, e)

                    # Save intermediary model variables
                    if config.save_model and e % max(1, int(config.num_epoch / 5)) == 0 and e != 0:
                        save_path = saver.save(sess, "save/tmp.ckpt", global_step=e)
                        print("\n Model saved in file: %s" % save_path)
                    e += 1
                print("\n Training COMPLETED! for one time slot")
                for i in range(len_sevice):#service chain lenght
                    # Constraint
                    if avail_nodeCap_Memory[positionarray[rewardarray.index(min(rewardarray))][0][i]] >= vnfavamemory[i] and avail_nodeCap_Core[positionarray[rewardarray.index(min(rewardarray))][0][i]] >= vnfresource[i]:

                        n = positionarray[rewardarray.index(min(rewardarray))][0][i]
                        v = i
                        pre_placement[n, v] = 1
                        m = np.where(dv_prev_array[:, v] == 1)
                        avail_nodeCap_Memory[n] = avail_nodeCap_Memory[n] - vnfavamemory[v]
                        avail_nodeCap_Memory[m[0][0]] = avail_nodeCap_Memory[m[0][0]] + vnfavamemory[v]
                        avail_nodeCap_Core[n] = avail_nodeCap_Core[n] - vnfresource[v]
                        avail_nodeCap_Core[m[0][0]] = avail_nodeCap_Core[m[0][0]] + vnfresource[v]

                    else:
                        v = i
                        n = np.where(dv_prev_array[:, v] == 1)
                        pre_placement[n[0][0], v] = dv_prev_array[n[0][0], v]
                outfile = open('BestPosition', 'wb')
                pic.dump(pre_placement, outfile)
                outfile.close()

                #New pre-placement Read
                infile = open('Bestposition', 'rb')
                dv_prev_array = pic.load(infile)
                infile.close()
                outfile = open('NodeCapMemory', 'wb')
                pic.dump(avail_nodeCap_Memory, outfile)
                outfile.close()
                outfile = open('NodeCapCore', 'wb')
                pic.dump(avail_nodeCap_Core, outfile)
                outfile.close()
                infile = open('NodeCapCore', 'rb')
                avail_nodeCap_Core = pic.load(infile)
                infile.close()
                infile = open('NodeCapMemory', 'rb')
                avail_nodeCap_Memory = pic.load(infile)
                infile.close()
                if config.save_model:
                    save_path = saver.save(sess, "save/tf_vnfMigration.ckpt")
                    print("\n Model saved in file: %s" % save_path)
