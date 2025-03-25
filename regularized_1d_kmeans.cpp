#include <vector>
#include <cassert>

#include "regularized_1d_kmeans.h"


class kmeans_wilber {
public:
    kmeans_wilber(double lambda, const npq::CostCalculator& cc, npq::id_t n);

    void wilber();
    std::vector<npq::id_t> smawk_inner(std::vector<npq::id_t>& columns, npq::id_t e, std::vector<npq::id_t>& rows);
    std::vector<double> smawk(npq::id_t i0, npq::id_t i1, npq::id_t j0, npq::id_t j1, std::vector<npq::id_t>& bl);
    double weight(npq::id_t i, npq::id_t j);
    double g(npq::id_t i, npq::id_t j);

    double lambda;  // regularization parameter
    std::vector<double> f;  // internal array for wilbers algorithm only updated in wilber
    std::vector<npq::id_t> bestleft;  //shortest path from wilber - only updated in wilbur
    const npq::CostCalculator* pCc;  // cluster cost data structure based on prefix sums
    npq::id_t n;  // data size
};


kmeans_wilber::kmeans_wilber(double lambda, const npq::CostCalculator& cc, npq::id_t n) :
    lambda{ lambda }, f(n + 1, 0.0), bestleft(n + 1, 0), pCc(&cc), n(n) {}

double kmeans_wilber::weight(npq::id_t i, npq::id_t j) {
    if (i >= j) return std::numeric_limits<double>::max();
    return (*pCc)(i, j - 1) + lambda;
}

double kmeans_wilber::g(npq::id_t i, npq::id_t j) {
    return f[i] + weight(i, j);
}

std::vector<npq::id_t> kmeans_wilber::smawk_inner(std::vector<npq::id_t>& columns, npq::id_t e, std::vector<npq::id_t>& rows) {
    // base case.
    npq::id_t n = columns.size();
    npq::id_t result_size = (n + e - 1) / e;
    if (rows.size() == 1) {
        // result is of length (n + e - 1)/e
        return std::vector<npq::id_t>(result_size, 0);
    }

    //reduce
    std::vector<npq::id_t> new_rows;
    std::vector<npq::id_t> translate;
    if (result_size < rows.size()) {
        for (npq::id_t i = 0; i < rows.size(); ++i) {
            // I1: forall j in [0..new_rows.size() - 2]: g(new_rows[j], columns[e*j]) < g(new_rows[j+1], columns[e*j]).
            // for (npq::id_t j = 1; j < new_rows.size(); ++j) {
            //     assert(g(new_rows[j-1], columns[e*(j-1)]) < g(new_rows[j], columns[e*(j-1)]));
            // }
            // I2: every column minima is either already in a row in new_rows OR
            //                         it is in rows[j] where j >= i.
            auto r = rows[i];
            //&& (new_rows.size() - 1 + (rows.size() - i)) >= result_size
            while (new_rows.size() &&
                g(r, columns[e * (new_rows.size() - 1)]) <= g(new_rows.back(), columns[e * (new_rows.size() - 1)])) {
                new_rows.pop_back();
                translate.pop_back();
            }
            if (e * new_rows.size() < n) { new_rows.push_back(r); translate.push_back(i); }
        }
    }
    else {
        new_rows = rows;
        for (npq::id_t i = 0; i < rows.size(); ++i) translate.push_back(i);
    }
    // assert(new_rows.size() <= result_size); // new_row.size() = ceil(n/e)
    // assert(new_rows.size());
    if (result_size == 1) {
        return std::vector<npq::id_t>{translate[0]};
    }
    //recurse
    std::vector<npq::id_t> column_minima_rec = smawk_inner(columns, 2 * e, new_rows);  // indexes in new_rows
    // assert(column_minima_rec.size() == ((result_size + 1)/ 2));
    std::vector<npq::id_t> column_minima; // indexes in rows

    //combine.
    column_minima.push_back(translate[column_minima_rec[0]]);
    for (npq::id_t i = 1; i < column_minima_rec.size(); ++i) {
        npq::id_t from = column_minima_rec[i - 1];  // index in new_rows
        npq::id_t to = column_minima_rec[i]; // index in new_rows
        npq::id_t new_column = (2 * i - 1); // 1, 3, 5..

        // assert(column_minima.size() == new_column);

        column_minima.push_back(from);
        for (npq::id_t r = from; r <= to; ++r) {
            if (g(new_rows[r], columns[new_column * e]) <= g(rows[column_minima[new_column]], columns[new_column * e])) {
                column_minima[new_column] = translate[r];
            }
        }
        column_minima.push_back(translate[to]);
    }
    // assert(column_minima.size() == result_size || column_minima.size() == result_size - 1);
    if (column_minima.size() < result_size) {
        // assert(column_minima.size() == result_size - 1);
        npq::id_t from = column_minima_rec.back();
        npq::id_t new_column = column_minima.size();

        column_minima.push_back(translate[from]);
        for (npq::id_t r = from; r < new_rows.size(); ++r) {
            if (g(new_rows[r], columns[new_column * e]) <= g(rows[column_minima[new_column]], columns[new_column * e])) {
                column_minima[new_column] = translate[r];
            }
        }
    }
    // assert(column_minima.size() == result_size);
    return column_minima;
}

std::vector<double> kmeans_wilber::smawk(npq::id_t i0, npq::id_t i1, npq::id_t j0, npq::id_t j1, std::vector<npq::id_t>& idxes) {
    std::vector<npq::id_t> rows, cols;
    for (npq::id_t i = i0; i <= i1; ++i) {
        rows.push_back(i);
    }
    for (npq::id_t j = j0; j <= j1; ++j) {
        cols.push_back(j);
    }
    std::vector<npq::id_t> column_minima = smawk_inner(cols, 1, rows); // indexes in rows.
    std::vector<double> res(column_minima.size());
    for (npq::id_t i = 0; i < res.size(); ++i) {
        res[i] = g(rows[column_minima[i]], cols[i]);
        idxes.push_back(rows[column_minima[i]]);
        assert(res[i] != std::numeric_limits<double>::max());
    }
    return res;
}

void kmeans_wilber::wilber() {
    //std::cout << "call " << name() << " with lambda=" << lambda << std::endl;
    f.resize(n + 1, 0);
    bestleft.resize(n + 1, 0);
    f[0] = 0;
    npq::id_t c = 0, r = 0;

    while (c < n) {
        //std::cout << "hello" << std::endl;
        // step 1
        npq::id_t p = std::min(2 * c - r + 1, n);
        // step 2
        {
            std::vector<npq::id_t> bl;
            std::vector<double> column_minima = smawk(r, c, c + 1, p, bl);
            for (npq::id_t j = c + 1; j <= p; ++j) {
                f[j] = column_minima[j - (c + 1)];
                bestleft[j] = bl[j - (c + 1)];
            }
        }
        // step 3
        if (c + 1 <= p - 1) {
            std::vector<npq::id_t> bl;
            std::vector<double> H = smawk(c + 1, p - 1, c + 2, p, bl);
            // step 4
            npq::id_t j0 = p + 1;
            for (npq::id_t j = p; j >= c + 2; --j) {
                if (H[j - (c + 2)] < f[j]) j0 = j;
            }
            //step 5
            if (j0 == p + 1) {
                c = p;
            }
            else {
                f[j0] = H[j0 - (c + 2)];
                bestleft[j0] = bl[j0 - (c + 2)];
                r = c + 1;
                c = j0;
            }
        }
        else {
            c = p;
        }
    }
}


std::vector<npq::id_t> npq::computeRegularized1DKMeans(double lambda, const npq::CostCalculator& cc, npq::id_t n)
{
    kmeans_wilber kw{ lambda, cc, n };
    kw.wilber();

    std::vector<npq::id_t> path;
	path.reserve(n);
    npq::id_t m = n;
    while (m != 0) {
        npq::id_t prev = kw.bestleft[m];
        path.push_back(prev);
        m = prev;
    }
	std::reverse(path.begin(), path.end());

    return path;
}
