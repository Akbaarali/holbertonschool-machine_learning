    # bins every 10 units (0-100)
    bins = np.arange(0, 101, 10)

    plt.hist(student_grades, bins=bins, edgecolor='black')

    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')

    plt.xticks(bins)
    plt.xlim(0, 100)

    plt.show()
