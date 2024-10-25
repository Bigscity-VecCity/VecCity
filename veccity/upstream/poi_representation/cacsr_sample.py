import pandas as pd
import numpy as np
from veccity.upstream.poi_representation.utils import get_delta, split_sampleSeq2sessions, splitSeq_basedonSessions, get_relativeTime
import copy
class cacsr_sample(object):

    def __init__(self, venueId2lid, SS_distance=None, SS_gaussian_distance=None, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2,
                max_his_period_days=28, max_merge_seconds_limit=1200, max_delta_mins=720, min_session_mins=1440):
        self.SS_distance = SS_distance
        self.SS_gaussian_distance = SS_gaussian_distance
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.max_his_period_days = max_his_period_days  
        self.max_merge_seconds_limit = max_merge_seconds_limit  
        self.max_delta_mins = max_delta_mins  
        self.min_session_mins = min_session_mins  
        self.userId2uid = dict()  
        self.venueId2lid = venueId2lid
        self.us = []  
        self.vs = []
        self.user_cnt = 0
        self.venue_cnt = len(venueId2lid)
        self.user_lidfreq = []  # (nb_user, nb_location)
        # train samples
        self.trainX_sample_cnt = 0
        self.trainX_duration2first = []
        self.trainX_last_distances = []
        self.trainX_target_lengths = []
        self.trainX_arrival_times = []
        self.trainX_session_arrival_times = []
        self.trainX_local_weekdays = []
        self.trainX_session_local_weekdays = []
        self.trainX_local_hours = []
        self.trainX_session_local_hours = []
        self.trainX_local_mins = []
        self.trainX_session_local_mins = []
        self.trainX_delta_times = []
        self.trainX_session_delta_times = []
        self.trainX_locations = []
        self.trainX_session_locations = []
        self.trainX_lengths = []
        self.trainY_arrival_times = []
        self.trainY_delta_times = []
        self.trainY_locations = []
        self.trainX_session_num = []
        self.trainX_session_lengths = []
        self.trainX_users = []
        # val samples
        self.valX_sample_cnt = 0
        self.valX_duration2first = []
        self.valX_last_distances = []
        self.valX_target_lengths = []
        self.valX_arrival_times = []
        self.valX_session_arrival_times = []
        self.valX_local_weekdays = []
        self.valX_session_local_weekdays = []
        self.valX_local_hours = []
        self.valX_session_local_hours = []
        self.valX_local_mins = []
        self.valX_session_local_mins = []
        self.valX_delta_times = []
        self.valX_session_delta_times = []
        self.valX_locations = []
        self.valX_session_locations = []
        self.valX_lengths = []
        self.valY_arrival_times = []
        self.valY_delta_times = []
        self.valY_locations = []
        self.valX_session_num = []
        self.valX_session_lengths = []
        self.valX_users = []
        # test samples
        self.testX_sample_cnt = 0
        self.testX_duration2first = []
        self.testX_last_distances = []
        self.testX_target_lengths = []
        self.testX_arrival_times = []
        self.testX_session_arrival_times = []
        self.testX_local_weekdays = []
        self.testX_session_local_weekdays = []
        self.testX_local_hours = []
        self.testX_session_local_hours = []
        self.testX_local_mins = []
        self.testX_session_local_mins = []
        self.testX_delta_times = []
        self.testX_session_delta_times = []
        self.testX_locations = []
        self.testX_session_locations = []
        self.testX_lengths = []
        self.testY_arrival_times = []
        self.testY_delta_times = []
        self.testY_locations = []
        self.testX_session_num = []
        self.testX_session_lengths = []
        self.testX_users = []

    def deal_cluster_sequence_for_each_user(self, sequence): 
 
        drops = 0
        for index, row in sequence.iterrows():
            try:
                dis = 1
                while (sequence.loc[index, 'venueId'] == sequence.loc[index + dis, 'venueId']) and ((sequence.loc[index + dis, 'timestamp'] - sequence.loc[index, 'timestamp']) < self.max_merge_seconds_limit):
                    try:
                        sequence = sequence.drop(index + dis) 
                        dis = dis + 1
                        drops += 1
                    except KeyError:
                        dis = dis + 1
                        continue
            except KeyError:
                continue
        return sequence, drops

    def constructing(self, dataset, uid, session_based_lengths, sessions, duration2first, session_based_arrival_times, session_based_locations, session_based_delta_times, session_based_local_weekdays, session_based_local_hours, session_based_local_mins, sum_sess, all_arrival_times, all_locations, all_delta_times, all_local_weekdays, all_local_hours, all_local_mins):
        for sid in sessions:    
            n = len(session_based_arrival_times[sid])
            x = [i for i in range(n - 1)]
            y = [i for i in range(1, n)] 

            target_loc = [session_based_locations[sid][i] for i in y]
            target_tim = [session_based_arrival_times[sid][i] for i in y]
            target_delta_tim = [session_based_delta_times[sid][i] for i in y]

            cur_tim = [session_based_arrival_times[sid][i] for i in x]
            cur_loc = [session_based_locations[sid][i] for i in x]
            cur_delta_tim = [session_based_delta_times[sid][i] for i in x]
            cur_day = [session_based_local_weekdays[sid][i] for i in x]
            cur_hour = [session_based_local_hours[sid][i] for i in x]
            cur_minute = [session_based_local_mins[sid][i] for i in x]

            last_point_distances = [self.SS_gaussian_distance[lid] for lid in cur_loc] 

            self.us.append(cur_loc)
            self.vs.append(target_loc)

            history_tim = []
            history_loc = []
            history_delta_tim = []
            history_day = []
            history_hour = []
            history_minute = []

            session_based_history_tim = []
            session_based_history_loc = []
            session_based_history_delta_tim = []
            session_based_history_day = []
            session_based_history_hour = []
            session_based_history_minute = []

            st = 0 
            if sid > 0: 
                
                for i in range(0, sid):
                    if((target_tim[0] - session_based_arrival_times[i][0]).days < self.max_his_period_days):
                        st = i
                        break
                
                if st == 0: 
                    history_tim = all_arrival_times[ : sum_sess[sid - 1]]
                    history_loc = all_locations[ : sum_sess[sid - 1]]
                    history_delta_tim = all_delta_times[ : sum_sess[sid - 1]]
                    history_day = all_local_weekdays[ : sum_sess[sid - 1]]
                    history_hour = all_local_hours[ : sum_sess[sid - 1]]
                    history_minute = all_local_mins[ : sum_sess[sid - 1]]
                else:
                    history_tim = all_arrival_times[sum_sess[st - 1] : sum_sess[sid - 1]] 
                    history_loc = all_locations[sum_sess[st - 1] : sum_sess[sid - 1]]
                    history_delta_tim = all_delta_times[sum_sess[st - 1] : sum_sess[sid - 1]]
                    history_day = all_local_weekdays[sum_sess[st - 1] : sum_sess[sid - 1]]
                    history_hour = all_local_hours[sum_sess[st - 1] : sum_sess[sid - 1]]
                    history_minute = all_local_mins[sum_sess[st - 1] : sum_sess[sid - 1]]

                session_based_history_tim = session_based_arrival_times[st : sid] 
                session_based_history_loc = session_based_locations[st : sid]
                session_based_history_delta_tim = session_based_delta_times[st : sid]
                session_based_history_day = session_based_local_weekdays[st : sid]
                session_based_history_hour = session_based_local_hours[st : sid]
                session_based_history_minute = session_based_local_mins[st : sid]

            length = sum_sess[sid] 
            if st == 0: 
                sessions_duration2first = duration2first[ : (sum_sess[sid] - 1)] 
            else:
                length -= sum_sess[st - 1]
                sessions_duration2first = list(np.array(duration2first[sum_sess[st - 1] : (sum_sess[sid] - 1)]) - duration2first[sum_sess[st - 1]]) 

            all_tim = history_tim + (cur_tim)
            all_loc = history_loc + (cur_loc)
            all_delta_tim = history_delta_tim + (cur_delta_tim)
            all_day = history_day + (cur_day)
            all_hour = history_hour + (cur_hour)
            all_minute = history_minute + (cur_minute)
            
            all_session_based_tim = copy.deepcopy(session_based_history_tim) 
            all_session_based_tim.append(cur_tim) 
            all_session_based_loc = copy.deepcopy(session_based_history_loc)
            all_session_based_loc.append(cur_loc)
            all_session_based_delta_tim = copy.deepcopy(session_based_history_delta_tim)
            all_session_based_delta_tim.append(cur_delta_tim)
            all_session_based_day = copy.deepcopy(session_based_history_day)
            all_session_based_day.append(cur_day)
            all_session_based_hour = copy.deepcopy(session_based_history_hour)
            all_session_based_hour.append(cur_hour)
            all_session_based_minute = copy.deepcopy(session_based_history_minute)
            all_session_based_minute.append(cur_minute)

            #if dataset == 'train':
            #print(f"generate {n - 1} train samples", flush=True)
            self.trainX_sample_cnt += n - 1
            self.trainX_target_lengths.append(n - 1) 
            self.trainX_arrival_times.append(all_tim)
            self.trainX_session_arrival_times.append(all_session_based_tim)
            self.trainX_local_weekdays.append(all_day)
            self.trainX_session_local_weekdays.append(all_session_based_day)
            self.trainX_local_hours.append(all_hour)
            self.trainX_session_local_hours.append(all_session_based_hour)
            self.trainX_local_mins.append(all_minute)
            self.trainX_session_local_mins.append(all_session_based_minute)
            self.trainX_delta_times.append(all_delta_tim)
            self.trainX_session_delta_times.append(all_session_based_delta_tim)
            self.trainX_locations.append(all_loc)
            self.trainX_session_locations.append(all_session_based_loc)
            self.trainX_lengths.append(length)
            self.trainY_arrival_times.append(target_tim)
            self.trainY_delta_times.append(target_delta_tim)
            self.trainY_locations.append(target_loc)
            self.trainX_session_num.append(sid - st + 1)  #
            self.trainX_session_lengths.append(session_based_lengths[st:sid + 1])  
            self.trainX_users.append(uid)
            self.trainX_duration2first.append(sessions_duration2first)
            self.trainX_last_distances.append(last_point_distances)
            # elif dataset == 'val':
            #     print(f"generate {n - 1} val samples", flush=True)
            #     self.valX_sample_cnt += n - 1
            #     self.valX_target_lengths.append(n - 1)
            #     self.valX_arrival_times.append(all_tim)
            #     self.valX_session_arrival_times.append(all_session_based_tim)
            #     self.valX_local_weekdays.append(all_day)
            #     self.valX_session_local_weekdays.append(all_session_based_day)
            #     self.valX_local_hours.append(all_hour)
            #     self.valX_session_local_hours.append(all_session_based_hour)
            #     self.valX_local_mins.append(all_minute)
            #     self.valX_session_local_mins.append(all_session_based_minute)
            #     self.valX_delta_times.append(all_delta_tim)
            #     self.valX_session_delta_times.append(all_session_based_delta_tim)
            #     self.valX_locations.append(all_loc)
            #     self.valX_session_locations.append(all_session_based_loc)
            #     self.valX_lengths.append(length)
            #     self.valY_arrival_times.append(target_tim)
            #     self.valY_delta_times.append(target_delta_tim)
            #     self.valY_locations.append(target_loc)
            #     self.valX_session_num.append(sid - st + 1)  
            #     self.valX_session_lengths.append(session_based_lengths[st:sid])  
            #     self.valX_users.append(uid)
            #     self.valX_duration2first.append(sessions_duration2first)
            #     self.valX_last_distances.append(last_point_distances)
            # else:
            #     print(f"generate {n - 1} test samples", flush=True)
            #     self.testX_sample_cnt += n - 1
            #     self.testX_target_lengths.append(n - 1)
            #     self.testX_arrival_times.append(all_tim)
            #     self.testX_session_arrival_times.append(all_session_based_tim)
            #     self.testX_local_weekdays.append(all_day)
            #     self.testX_session_local_weekdays.append(all_session_based_day)
            #     self.testX_local_hours.append(all_hour)
            #     self.testX_session_local_hours.append(all_session_based_hour)
            #     self.testX_local_mins.append(all_minute)
            #     self.testX_session_local_mins.append(all_session_based_minute)
            #     self.testX_delta_times.append(all_delta_tim)
            #     self.testX_session_delta_times.append(all_session_based_delta_tim)
            #     self.testX_locations.append(all_loc)
            #     self.testX_session_locations.append(all_session_based_loc)
            #     self.testX_lengths.append(length)
            #     self.testY_arrival_times.append(target_tim)
            #     self.testY_delta_times.append(target_delta_tim)
            #     self.testY_locations.append(target_loc)
            #     self.testX_session_num.append(sid - st + 1)  
            #     self.testX_session_lengths.append(session_based_lengths[st:sid])  
            #     self.testX_users.append(uid)
            #     self.testX_duration2first.append(sessions_duration2first)
            #     self.testX_last_distances.append(last_point_distances)

    def construct_sample_seq2seq(self, checkins, uid):

        arrival_times = list(checkins['local time'][:])
        arrival_local_weekdays = list(checkins['local weekday'].values)
        arrival_local_hours = list(checkins['local hour'].values)
        arrival_local_mins = list(checkins['local minute'].values)
        # locations = [self.venueId2lid[lid] for lid in checkins['venueId'].values]
        locations = list(checkins['lid'].values)
        # print("length is ", len(checkins))
        # print("arrival_times: ", arrival_times[:10])
        # print("arrival_local_days: ", arrival_local_weekdays[:10])
        # print("arrival_local_hours: ", arrival_local_hours[:10])
        # print("arrival_local_mins: ", arrival_local_mins[:10])
        # print("locations: ", locations[:10])

        sessions = []
        delta_arrival_times = get_delta(arrival_times)
        #print("delta_arrival_times: ", delta_arrival_times)
        session_based_delta_times, split_index, session_based_lengths = split_sampleSeq2sessions(delta_arrival_times, self.min_session_mins)
        # print("split_index: ", split_index) 
        #print("session data: ", sum(session_based_lengths), ", ", len(session_based_lengths))
        if sum(session_based_lengths) >= 5 and len(session_based_lengths) >= 3: 
            session_based_arrival_times = splitSeq_basedonSessions(arrival_times, split_index)
            session_based_local_weekdays = splitSeq_basedonSessions(arrival_local_weekdays, split_index)
            session_based_local_hours = splitSeq_basedonSessions(arrival_local_hours, split_index)
            session_based_local_mins = splitSeq_basedonSessions(arrival_local_mins, split_index)
            session_based_locations = splitSeq_basedonSessions(locations, split_index)

            duration2first = get_relativeTime([i for k in session_based_arrival_times for i in k]) 
            all_arrival_times = ([i for k in session_based_arrival_times for i in k])
            all_local_weekdays = ([i for k in session_based_local_weekdays for i in k])
            all_local_hours = ([i for k in session_based_local_hours for i in k])
            all_local_mins = ([i for k in session_based_local_mins for i in k])
            all_locations = ([i for k in session_based_locations for i in k])
            all_delta_times = get_delta(all_arrival_times)

            # print("all_arrival_times data: ", all_arrival_times)
            # print("all_locations data: ", all_locations)
            # print("session_based_arrival_times data: ", session_based_arrival_times)
            # print("session_based_locations data: ", session_based_locations)
            
            sessions_length = len(session_based_lengths)
            sum_sess = {}
            sum_sess[0] = session_based_lengths[0]
            for i in range(1, sessions_length): 
                sum_sess[i] = sum_sess[i - 1] + session_based_lengths[i]; 


            all_length = sum_sess[sessions_length - 1]
            is_train = True
            train_cnt = 0
            val_cnt = 0
            delta_ratio = 0.05 
            for i in range(0, sessions_length):
                if is_train and sum_sess[i] > (all_length * (self.train_ratio - delta_ratio)):
                    train_cnt = i + 1
                    is_train = False
                elif sum_sess[i] > all_length * (self.val_ratio + self.train_ratio - delta_ratio):
                    val_cnt = i + 1  - train_cnt 
                    break
            test_cnt  = sessions_length - train_cnt - val_cnt
            if train_cnt <= 0 or val_cnt <= 0 or test_cnt <= 0: 
                return False

            train_sessions = [i for i  in range(train_cnt)] 
            val_sessions = [i for i in range(train_cnt, val_cnt + train_cnt)]
            test_sessions = [i for i in range(val_cnt + train_cnt, sessions_length)]

            self.constructing("train", uid, session_based_lengths, train_sessions, duration2first, session_based_arrival_times, session_based_locations, session_based_delta_times, session_based_local_weekdays, session_based_local_hours, session_based_local_mins, sum_sess, all_arrival_times, all_locations, all_delta_times, all_local_weekdays, all_local_hours, all_local_mins)
            self.constructing("val", uid, session_based_lengths, val_sessions, duration2first, session_based_arrival_times, session_based_locations, session_based_delta_times, session_based_local_weekdays, session_based_local_hours, session_based_local_mins, sum_sess, all_arrival_times, all_locations, all_delta_times, all_local_weekdays, all_local_hours, all_local_mins)
            self.constructing("test", uid, session_based_lengths, test_sessions, duration2first, session_based_arrival_times, session_based_locations, session_based_delta_times, session_based_local_weekdays, session_based_local_hours, session_based_local_mins, sum_sess, all_arrival_times, all_locations, all_delta_times, all_local_weekdays, all_local_hours, all_local_mins)

            return True
        else:
            return False