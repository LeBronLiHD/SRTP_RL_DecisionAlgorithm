import algorithm
import numpy as np


def matrix_to_move(matrix1, matrix2):
    delta_matrix = matrix2 - matrix1

    move = [0, 0, 0, 0]

    for i in range(10):
        for j in range(10):
            if delta_matrix[i][j] < 0:
                if matrix2[i][j] == 0:
                    move[0] = i
                    move[1] = j
                else:
                    move[2] = i
                    move[3] = j
            if delta_matrix[i][j] > 0:
                move[2] = i
                move[3] = j

    return move


if __name__ == "__main__":
    string1 = "red-conn<1>5,00;0,01;0,02;7,03;0,04;0,05;14,06;0,07;0,08;12,09;4,10;0,11;6,12;0,13;0,14;0,15;0,16;13," \
              "17;0,18;11,19;3,20;0,21;0,22;7,23;0,24;0,25;14,26;0,27;0,28;10,29;2,30;0,31;0,32;0,33;0,34;0,35;0," \
              "36;0,37;0,38;9,39;1,40;0,41;0,42;7,43;0,44;0,45;14,46;0,47;0,48;8,49;2,50;0,51;0,52;0,53;0,54;0,55;0," \
              "56;0,57;0,58;9,59;3,60;0,61;0,62;7,63;0,64;0,65;14,66;0,67;0,68;10,69;4,70;0,71;6,72;0,73;0,74;0,75;0," \
              "76;13,77;0,78;11,79;5,80;0,81;0,82;7,83;0,84;0,85;14,86;0,87;0,88;12,89;"
    string2 = "red-conn<1>5,00;0,01;0,02;7,03;0,04;0,05;14,06;0,07;0,08;12,09;4,10;0,11;6,12;0,13;0,14;0,15;0,16;13," \
              "17;0,18;11,19;3,20;0,21;0,22;7,23;0,24;0,25;14,26;0,27;0,28;10,29;2,30;0,31;0,32;0,33;0,34;0,35;0," \
              "36;0,37;0,38;9,39;1,40;0,41;0,42;7,43;0,44;0,45;14,46;0,47;0,48;8,49;2,50;0,51;0,52;0,53;0,54;0,55;0," \
              "56;0,57;0,58;9,59;3,60;0,61;0,62;7,63;0,64;0,65;14,66;0,67;0,68;10,69;4,70;0,71;6,72;0,73;0,74;0,75;0," \
              "76;13,77;0,78;11,79;5,80;0,81;0,82;7,83;0,84;0,85;14,86;0,87;0,88;12,89;"

    matrix1, turn1 = algorithm.string2matrix(string1)
    matrix2, turn2 = algorithm.string2matrix(string2)
    move = matrix_to_move(np.array(matrix1), np.array(matrix2))
    print(move)

    exit()
