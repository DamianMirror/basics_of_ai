'''
Spam
Limits: 3 sec., 512 MiB
Given a set of emails, some of which are classified as normal, and others as spam. Then, new emails are given for which you need to calculate the probability that they are spam. The probability is calculated based on the occurrence of words that were in the classified emails. Words are independent of each other.
Input
The first line contains three integers N, M, and K - the number of normal emails, the number of spam emails, and the number of new emails.
The next N lines contain texts of normal emails (one email per line).
The next M lines contain texts of spam emails (one email per line).
The next K lines contain texts of new emails (one email per line).
Emails contain only words with lowercase English letters and spaces between them.
Output
Output K lines. In the i-th line, output the probability that the i-th email is spam.
The absolute or relative error of the results should not exceed 0.0001.
Constraints

0 ≤ N ≤ 1000
0 ≤ M ≤ 1000
1 ≤ N + M
1 ≤ K ≤ 1000
Total length of texts is no more than 10,000,000 characters.
One email contains no more than 100,000 characters.
'''

from collections import defaultdict

def main():
    N, M, K = map(int, input().split())

    normal_emails = [input().strip() for _ in range(N)]
    spam_emails = [input().strip() for _ in range(M)]
    new_emails = [input().strip() for _ in range(K)]

    spam_probability = M / (N + M)
    normal_probability = N / (N + M)

    word_counts_normal = defaultdict(int)
    word_counts_spam = defaultdict(int)

    for email in normal_emails:
        words = set(email.split())  # unique words only
        for word in words:
            word_counts_normal[word] += 1

    for email in spam_emails:
        words = set(email.split())  # unique words only
        for word in words:
            word_counts_spam[word] += 1


    for email in new_emails:
        words = email.split()
        prob_spam = spam_probability
        prob_normal = normal_probability

        for word in words:
            prob_word_given_spam = word_counts_spam[word] / M
            prob_word_given_normal = word_counts_normal[word] / N

            if prob_word_given_spam == 0:
                prob_word_given_spam = 1 / (M * 10**6)
            if prob_word_given_normal == 0:
                prob_word_given_normal = 1 / (N * 10**6)

            prob_spam *= prob_word_given_spam
            prob_normal *= prob_word_given_normal

        #normalization
        prob_normal = prob_normal / prob_spam + prob_normal
        prob_spam = prob_spam / prob_spam + prob_normal

        total_prob = prob_spam + prob_normal
        if total_prob == 0:
            print(0.0)
        else:
            print(prob_spam / total_prob)



if __name__ == "__main__":
    main()