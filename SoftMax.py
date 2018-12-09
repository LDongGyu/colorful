#-*- coding:utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import csv

# 전처리

cocktail_list = np.loadtxt('cocktail.csv', delimiter=',',dtype=object) # 칵테일 리스트
cocktail_list = cocktail_list[:, 6]

cocktail_info_taste = {1: "깔끔", 2: "단맛", 3:"상큼", 4:"새콤", 5:"신맛", 6:"쓴맛", 7:"짠맛"}
cocktail_info_alchol = {1: "맥주정도", 2:"소주정도", 3:"양주정도", 4:"제일 쌘거"}
cocktail_info_soda = {0: "없음", 1:"있음"}
cocktail_info_mouthfeel = {0:"물", 1:"약간 걸쭉", 2:"걸쭉", 3:"부드러움"}
cocktail_info_base = {0:'데킬라', 1:"럼", 2:"리큐르", 3:"맥주", 4:"보드카", 5:"브랜디", 6:"샴페인", 7:"와인", 8:"위스키", 9:"진"}

while True:
    type = int(input("type을 설정해주세요 (1. 입맛으로 추천, 2. 비슷한 유저로 추천, 3. 칵테일 정보보기, 나머지 입력은 종료) : "))

    if type == 1: # 내 입맛으로 추천

        taste = int(input("맛 : 1.깔끔 2.단맛 3.상큼 4.새콤 5.신맛 6.쓴맛 7.짠맛 "))
        alchol = int(input("도수 : 1.맥주정도 2.소주정도 3.양주정도 4.그 이상 "))
        soda = int(input("탄산 : 0.없음 1.있음 "))
        mouthfeel = int(input("식감 : 0.물 1.약간 걸쭉 2.걸쭉 3.부드러움 "))
        base = int(input("기주 : 0.데킬라 1.럼 2.리큐르 3.맥주 4.보드카 5.브랜디 6.샴페인 7.와인 8.위스키 9.진 "))

        cocktail_data = np.loadtxt('cocktail.csv', delimiter=',', dtype=np.object)
        cocktail_x_data = cocktail_data[:, 0:5]
        cocktail_y_data = cocktail_data[:, 7:]

        cocktail_X = tf.placeholder(tf.float32, shape=[None,5])
        cocktail_Y = tf.placeholder(tf.float32, shape=[None,60])

        nb_classes = 60

        W = tf.Variable(tf.random_normal([5,nb_classes]),name='weight')
        b = tf.Variable(tf.random_normal([nb_classes]),name='bias')

        cocktail_hypothesis = tf.nn.softmax(tf.matmul(cocktail_X,W)+b)
        cocktail_cost = tf.reduce_mean(-tf.reduce_sum(cocktail_Y*tf.log(cocktail_hypothesis),axis=1))
        cocktail_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cocktail_cost)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print("계산중...")
            for step in range(20001):
                sess.run(cocktail_optimizer, feed_dict={cocktail_X:cocktail_x_data,cocktail_Y:cocktail_y_data})
                # if step % 400 == 0:
                #     print(step, sess.run(cocktail_cost, feed_dict={cocktail_X:cocktail_x_data,cocktail_Y:cocktail_y_data}))
            recommend1 = sess.run(cocktail_hypothesis, feed_dict={cocktail_X: [[taste, alchol, soda ,mouthfeel, base]]})
            print("cocktail recommend : ",cocktail_list[sess.run(tf.argmax(recommend1,1))])

            f = open('cocktail.csv', 'a', encoding='utf-8', newline='')
            wr = csv.writer(f)
            write_data = []
            write_data.append(taste)
            write_data.append(alchol)
            write_data.append(soda)
            write_data.append(mouthfeel)
            write_data.append(base)
            write_data.append(int(sess.run(tf.argmax(recommend1,1)+1)))
            write_data.append(0)

            for i in range(8,68):
                if i != int(sess.run(tf.argmax(recommend1,1)+1)+7):
                    write_data.append(0)
                else:
                    write_data.append(1)

            wr.writerow(write_data)
            f.close()
        # os.system('CLS')
    elif type== 2: # 비슷한 유저로 추천
        j = 1
        for i in range(0,60):
            if j%10 != 0:   # 10개 나열
                print(j,".",cocktail_list[i]," ", end='')
            else: # 10번째에서 개행
                print(j, ".", cocktail_list[i], " ")
            j+=1

        print("좋아하는 술을 번호로 입력해주세요!")

        drink1 = (float(input("1번째 선호하는 술 : "))/10)
        drink2 = (float(input("2번째 선호하는 술 : "))/10)
        drink3 = (float(input("3번째 선호하는 술 : "))/10)
        drink4 = (float(input("4번째 선호하는 술 : "))/10)
        drink5 = (float(input("5번째 선호하는 술 : "))/10)

        user_data = np.loadtxt('user.csv', delimiter=',', dtype=np.float32)
        user_x_data = user_data[:, 0:5]
        user_y_data = user_data[:, 6:]

        user_X = tf.placeholder(tf.float32, shape=[None,5])
        user_Y = tf.placeholder(tf.float32, shape=[None,60])

        nb_classes = 60

        user_W = tf.Variable(tf.random_normal([5,nb_classes]),name='weight')
        user_b = tf.Variable(tf.random_normal([nb_classes]),name='bias')

        user_hypothesis = tf.nn.softmax(tf.matmul(user_X,user_W)+user_b)
        user_cost = tf.reduce_mean(-tf.reduce_sum(user_Y*tf.log(user_hypothesis),axis=1))
        user_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(user_cost)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print("계산중...")
            for step in range(50001):
                sess.run(user_optimizer, feed_dict={user_X:user_x_data,user_Y:user_y_data})
                # if step % 400 == 0:
                #     print(step, sess.run(user_cost, feed_dict={user_X:user_x_data,user_Y:user_y_data}))
            recommend2 = sess.run(user_hypothesis, feed_dict={user_X: [[drink1, drink2, drink3 ,drink4, drink5]]})
            print("user recommend : ", cocktail_list[sess.run(tf.argmax(recommend2,1))])

            f = open('user.csv', 'a', encoding='utf-8', newline='')
            wr = csv.writer(f)
            write_data = []
            write_data.append(drink1)
            write_data.append(drink2)
            write_data.append(drink3)
            write_data.append(drink4)
            write_data.append(drink5)
            write_data.append(int(sess.run(tf.argmax(recommend2,1)+1)))

            for i in range(7,67):
                if i != int(sess.run(tf.argmax(recommend2,1)+1)+6):
                    write_data.append(0)
                else:
                    write_data.append(1)

            wr.writerow(write_data)
            f.close()

    elif type==3: # 칵테일 정보
        j = 1
        for i in range(0,60):
            if j%10 != 0:   # 10개 나열
                print(j,".",cocktail_list[i]," ", end='')
            else: # 10번째에서 개행
                print(j,".",cocktail_list[i], " ")
            j+=1
        cocktail_num = int(input("보고싶은 칵테일의 번호를 입력해주세요 : "))

        cocktail_data = np.loadtxt('cocktail.csv', delimiter=',', dtype=np.object)
        cocktail_x_data = cocktail_data[:, 0:5]

        print("이름 : ", cocktail_list[cocktail_num-1])
        print("맛 : ", cocktail_info_taste.get(int(cocktail_x_data[int(cocktail_num-1)][0])))
        print("도수 : ", cocktail_info_alchol.get(int(cocktail_x_data[int(cocktail_num-1)][1])))
        print("탄산 : ", cocktail_info_soda.get(int(cocktail_x_data[int(cocktail_num-1)][2])))
        print("식감 : ", cocktail_info_mouthfeel.get(int(cocktail_x_data[int(cocktail_num-1)][3])))
        print("기주 : ", cocktail_info_base.get(int(cocktail_x_data[int(cocktail_num-1)][4])))

    else: # 종료
        break