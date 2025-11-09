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

    # Handle edge cases
    if M == 0:
        for _ in range(K):
            print(0.0)
        return
    if N == 0:
        for _ in range(K):
            print(1.0)
        return

    # Prior probabilities
    prior_spam = M / (N + M)
    prior_normal = N / (N + M)

    word_count_normal = defaultdict(int)
    word_count_spam = defaultdict(int)

    for email in normal_emails:
        words = email.split()
        for word in words:
            word_count_normal[word] += 1

    for email in spam_emails:
        words = email.split()
        for word in words:
            word_count_spam[word] += 1

    # Total word counts
    total_spam_words = sum(word_count_spam.values())
    total_normal_words = sum(word_count_normal.values())

    # Process each new email
    for email in new_emails:
        words = email.split()

        prob_spam = prior_spam
        prob_normal = prior_normal

        for word in words:
            # Probability based on word frequency
            prob_spam *= word_count_spam[word] / total_spam_words
            prob_normal *= word_count_normal[word] / total_normal_words

        # Normalization
        total = prob_spam + prob_normal

        if total == 0:
            result = 0.0
        else:
            result = prob_spam / total

        print(result)


if __name__ == "__main__":
    main()