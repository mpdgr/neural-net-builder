#
# MATRIX MULTIPLICATION UTILS
#

# ------------------------------ product of two matrices -----------------------------
def matrix_product(m1, m2):
    # if any of the matrices is as a single vector(list) wrap it into a list to form a matrix
    if type(m1[0]) != list:
        m1 = [m1]
    if type(m2[0]) != list:
        m2 = [m2]

    if not equal_rows(m1) or not equal_rows(m2):
        print("matrix input incorrect")
        return

    if not cols_rows_match(m1, m2):
        print("number of columns of the first matrix must match the number of rows of the second matrix")
        return

    output = []

    columns = len(m2[0])
    rows = len(m1)

    for m in range(rows):
        output_row = []
        for n in range(columns):
            output_row.append(vector_dot_product(m1[m], extract_column(m2, n)))
        output.append(output_row)

    return output


# --------------------------------- transpose matrix ---------------------------------
def transpose_matrix(m):
    # if the matrix is as a single vector(list) wrap it into a list to form a matrix
    if type(m[0]) != list:
        m = [m]

    if not equal_rows(m):
        print("matrix input incorrect")
        return

    output = []

    cols = len(m[0])
    rows = len(m)

    for r in range(cols):
        output_row = []
        for c in range(rows):
            output_row.append(m[c][r])
        output.append(output_row)

    return output


# ---------------------------- dot product of two vectors ----------------------------
def vector_dot_product(v1, v2):
    if len(v1) != len(v2):
        print("vectors must be equal in length")
        return

    output = 0
    for i in range(len(v1)):
        output += v1[i] * v2[i]
    return output


# --------------------------- outer product of two vectors ---------------------------
def vector_outer_product(v1, v2):
    output = []
    for m in range(len(v1)):
        output_row = []
        for n in range(len(v2)):
            output_row.append(v1[m] * v2[n])
        output.append(output_row)
    return output


# ------------------ vector matrix row-wise/ element-wise product --------------------
def vector_matrix_row_wise_product(vector, matrix):
    # if the vectors is passed as a matrix (list of lists) unwrap it before processing
    if type(vector) == list and type(vector[0]) == list:
        vector = vector[0]

    if len(vector) != len(matrix[0]):
        print("vector length must equal matrix columns count")
        return

    output = []
    for i in range(len(matrix)):
        output.append(vector_element_wise_product(vector, matrix[i]))

    return output


# --------------------------- vector element-wise product ----------------------------
def vector_element_wise_product(v1, v2):
    # if any of the vectors is passed as a matrix (list of lists) unwrap it before processing
    if type(v1) == list and type(v1[0]) == list:
        v1 = v1[0]
    if type(v2) == list and type(v2[0]) == list:
        v2 = v2[0]

    if len(v1) != len(v2):
        print("vectors must be equal in length")
        return

    output = []
    for i in range(len(v1)):
        output.append(v1[i] * v2[i])

    return output


# --------------------------- vector product with a scalar ---------------------------
def vector_scalar_product(vector, scalar):
    output = []

    for i in range(len(vector)):
        output.append(vector[i] * scalar)

    return output


# --------------------------- matrix product with a scalar ---------------------------
def matrix_scalar_product(matrix, scalar):
    output = []

    for i in range(len(matrix)):
        output.append(vector_scalar_product(matrix[i], scalar))

    return output


# -------------------------------- subtract vectors ----------------------------------
def subtract_vectors(v1, v2):
    # if any of the vectors is passed as a matrix (list of lists) unwrap it before processing
    if type(v1) == list and type(v1[0]) == list:
        v1 = v1[0]
    if type(v2) == list and type(v2[0]) == list:
        v2 = v2[0]

    if len(v1) != len(v2):
        print("vectors must be equal in length")
        return

    output = []
    for i in range(len(v1)):
        output.append(v1[i] - v2[i])

    return output


# ------------------------------- subtract matrices ----------------------------------
def subtract_matrices(m1, m2):
    if not equal_size(m1, m2):
        print("matrices must be equal in size")
        return

    output = []
    for m in range(len(m1)):
        output_row = []
        for n in range(len(m1[0])):
            output_row.append(m1[m][n] - m2[m][n])
        output.append(output_row)

    return output


# ---------------------- subtract vector from matrix row-wise ------------------------
def subtract_vector_from_matrix_rows(vector, matrix):
    # if the vectors is passed as a matrix (list of lists) unwrap it before processing
    if type(vector) == list and type(vector[0]) == list:
        vector = vector[0]

    if len(vector) != len(matrix[0]):
        print("vectors must be equal in length")
        return

    output = []
    for i in range(len(matrix)):
        output.append(subtract_vectors(matrix[i], vector))

    return output


# -------------------------- extract matrix column to vector -------------------------
def extract_column(matrix, column_index):
    column = []
    for row in matrix:
        column.append(row[column_index])
    return column


# ----------------------------- matrix to row-major vector ----------------------------
def matrix_to_vector_row_major(matrix):
    vector = []
    for r in matrix:
        for item in r:
            vector.append(item)
    return vector


# -------------------- check if given list of lists forms a matrix -------------------
def equal_rows(matrix):
    columns = len(matrix[0])
    for m in matrix:
        if len(m) != columns:
            print("matrix lists must be equal in length")
            return False
    return True


# ----------------------- check if given matrices have same size ---------------------
def equal_size(m1, m2):
    return len(m1) == len(m2) and len(m1[0]) == len(m2[0])


# -- check if nr of cols of the first matrix equals nr of rows of the second matrix --
def cols_rows_match(m1, m2):
    return len(m1[0]) == len(m2)
