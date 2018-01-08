def getCurrentNumberOfPositives(test_acc, count_positive):
    if test_acc == 1.0:
        count_positive += 1
    return count_positive

def setAnswer(test_acc, queue_answer):
    queue_answer.append(test_acc)

def getPercentOfPositives(queue_answer):
    count = 0
    for ans in queue_answer:
        if ans == 1.0:
            count += 1
    return (count/len(queue_answer))*100

def getTruePositives():
    return False

def getTrueNegatives():
    return False

def getFalsePositives():
    return False

def getFalseNegatives():
    return False


