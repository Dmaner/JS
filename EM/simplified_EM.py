from random import randint, shuffle
import math

def GenerateCoin():
    """
    Generate three coins
    :return: Three Coins' lists
    """
    msg = "Coin {}: Positive probability : {:.2f}"
    dict = {}
    for alpha in "ABC":
        dict[alpha] = [randint(0, 1) for x in range(100)]
        print(msg.format(alpha, sum(dict[alpha])/100))

    return (dict[x] for x in "ABC")

def Genrating_observation_data(observation_length):
    """
    Generate observation list
    :param observation_length: the length of the observation
    :return: the final observation list
    """
    CoinA, CoinB, CoinC= GenerateCoin()
    results = []
    for i in range(observation_length):
        result = CoinA[randint(0, len(CoinA)-1)]
        if result == 1:
            results.append(CoinB[randint(0, len(CoinB)-1)])
        else:
            results.append(CoinC[randint(0, len(CoinC)-1)])

    return results

def calcul_u(pai, p, q, y):
    """
    Formula 9.5
    :param pai: probability of A
    :param p: probability of B
    :param q: probability of C
    :param y: the observation
    """
    B_A = pai*math.pow(p, y)*math.pow(1-p, 1-y)
    C_A = (1-pai)*math.pow(q, y)*math.pow(1-q, 1-y)
    return B_A/(B_A+C_A)

def EM_algorithon(num_epoch, test_length):
    """
    Implement the simplfied EM algorithom
    :param num_epoch: The number of train epoch
    :param test_list: The length of obsevation
    :return:
    """
    Y = Genrating_observation_data(test_length)
    PA, PB, PC = 0.5, 0.5, 0.5
    print("Begin....")
    msg = "Coin {}: Positive probability : {:.2f}"
    for i in range(num_epoch):

        # Step E
        PB_Y_list = [calcul_u(PA, PB, PC, y) for y in Y]

        # Step M
        PA = sum(PB_Y_list)/test_length
        PB = sum([x*y for x, y in zip(PB_Y_list, Y)])/sum(PB_Y_list)
        PC = sum([(1-x)*y for x, y in zip(PB_Y_list, Y)])/sum([1-x for x in PB_Y_list])

        # if i % 100==0:
        #     for alpha, p in zip("ABC", [PA, PB, PC]):
        #         print(msg.format(alpha, p))
    print("End")
    for alpha, p in zip("ABC", [PA, PB, PC]):
        print(msg.format(alpha, p))

EM_algorithon(100, 100)
