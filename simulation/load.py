import pandas as pd
import numpy as np
from scipy.stats import median_abs_deviation
from sklearn.metrics import adjusted_rand_score

class LoadData:
    def __init__(self,seed=0):
        self.seed = seed
        self.dataset_generator = np.random.default_rng(seed)
        
    def calculate_rand_score(self,true_changepoints, estimated_changepoints):
        """Compute the adjusted rand index between two sets of changepoints.

        Args:
        true_changepoints (List) : list of true change points.
        estimated_changepoints (List) : list of estimated change points.
        """
        true_changepoints = np.array(true_changepoints, dtype=np.int_)
        estimated_changepoints = np.array(estimated_changepoints, dtype=np.int_)

        y_true = np.zeros(true_changepoints[-1])
        for i, (start, stop) in enumerate(zip(true_changepoints[:-1], true_changepoints[1:])):
            y_true[start:stop] = i

        y_estimated = np.zeros(estimated_changepoints[-1])
        for i, (start, stop) in enumerate(zip(estimated_changepoints[:-1], estimated_changepoints[1:])):
            y_estimated[start:stop] = i

        return adjusted_rand_score(y_true, y_estimated)
    
    def normalize(self,X):
        pairwise_distances = X[1:, :] - X[:-1, :]
        # If the x_i are i.i.d. N(0, 1 \sigma), then the x_i - x_{i-1} are N(0, 2 \sigma).
        # Since sqrt(2) * MAD(x_i) is an estimator for \sigma, so is MAD(x_i - x_{i-1}).
        mad = median_abs_deviation(pairwise_distances, axis=0)
        mad[mad == 0] = 1
        return X / mad
    
    def _get_indices(self,segment_lengths, y,replace=True):
        
        segment_id = None
        indices = np.array([], dtype="int")
        value_counts = pd.value_counts(y)
        available_indices = np.ones(len(y), dtype=np.bool_)

        for segment_length in segment_lengths:
            available_segments = value_counts[lambda x: x.index != segment_id]
            if not replace:
                available_segments = available_segments[lambda x: (x >= segment_length).to_numpy()]

            if len(available_segments) == 0:
                raise ValueError("Not enough data.")

            segment_id = self.dataset_generator.choice(available_segments.index, 1)[0]

            new_indices = self.dataset_generator.choice(np.flatnonzero((y == segment_id) & available_indices),segment_length,replace=replace,)

            if not replace:
                value_counts.loc[segment_id] -= segment_length
                available_indices[new_indices] = False

            indices = np.concatenate([indices, new_indices])

        return indices
    
    def _exponential_segment_lengths(self,n_segments,n_observations,minimal_relative_segment_length=0.01):
        """Exponential segment lengths.

        Parameters
        ----------
        n_segments : int
            Number of segment lengths to be sampled.
        n_observations : int
            Number of observations in simulated time series. The sum of segment lengths
            returned will be equal to this.
        minimal_relative_segment_length : float, optional, default=0.01
            All segments will be at least `n * minimal_relative_segment_length` long.

        Returns
        -------
        numpy.ndarray
            Array with segment lengths.

        """

        expo = self.dataset_generator.exponential(scale=1, size=n_segments)
        expo = expo * (1 - minimal_relative_segment_length * n_segments) / expo.sum()
        expo = expo + minimal_relative_segment_length
        assert np.abs(expo.sum() - 1) < 1e-12
        assert np.min(expo) >= minimal_relative_segment_length
        return self._cascade_round(expo * n_observations)


    def _cascade_round(self,x):
        """Round floats in x to near integer, preserving their sum.

        Inspired by
        https://stackoverflow.com/questions/792460/how-to-round-floats-to-integers-while-preserving-their-sum

        """

        if np.abs(x.sum() - np.round(x.sum())) > 1e-8:
            raise ValueError("Values in x must sum to an integer value.")

        x_rounded = np.zeros(len(x), dtype=np.int_)
        remainder = 0

        for idx in range(len(x)):
            x_rounded[idx] = np.round(x[idx] + remainder)
            remainder += x[idx] - x_rounded[idx]

        assert np.abs(remainder) < 1e-8

        return x_rounded
    def get_data_from_csv(self,path,class_label="class",segment_sizes=None,minimal_relative_segment_length=0.01,normalize=True):
        data = pd.read_csv(path)
       

        data = data.reset_index(drop=True)

        value_counts = data[class_label].value_counts()

        if segment_sizes is None:
            if minimal_relative_segment_length is not None:
                value_counts = value_counts[lambda x: x / len(data) > minimal_relative_segment_length]

            idx = np.arange(len(value_counts))
            self.dataset_generator.shuffle(idx)
            value_counts = value_counts.iloc[idx]

            indices = np.array([], dtype=np.int_)

            for label, segment_size in value_counts.to_dict().items():
                indices = np.append(
                    indices,
                    self.dataset_generator.choice(data[lambda x: x[class_label] == label].index,segment_size,replace=False,),)

            segment_sizes = value_counts.to_numpy()
            changepoints = np.append([0], segment_sizes.cumsum())
            time_series = data.iloc[indices].drop(columns=class_label).to_numpy()
            if normalize:
                time_series = self.normalize(time_series)

            return time_series,list(changepoints)

        else:
            for _ in range(5):
                try:
                    indices = self._get_indices(segment_sizes, data[class_label].to_numpy())
                except ValueError:
                    continue

                changepoints = np.append([0], np.array(segment_sizes).cumsum())
                time_series = data.iloc[indices].drop(columns=class_label).to_numpy()
                if normalize:
                    time_series = self.normalize(time_series)

                return time_series,list(changepoints)

            raise ValueError("Not enough data")
    def get_data_from_csv_with_noise(self,path,class_label="class",signal_to_noise=1,n_observations=10000,n_segments=100,minimal_relative_segment_length=None):
        if minimal_relative_segment_length is None:
            minimal_relative_segment_length = 1 / n_segments / 10

        

        data = pd.read_csv(path)

        y = data[class_label].to_numpy()

        variances = (  # Compute variances separately for each class, take weighted mean.
            data.groupby(class_label).var() * data.groupby(class_label).count()
        ).sum(axis=0) / (data.shape[0] - data[class_label].nunique())
        X = (data.drop(columns=class_label) / variances.apply(np.sqrt)).to_numpy()

        segment_lengths = self._exponential_segment_lengths(n_segments, n_observations, minimal_relative_segment_length)
        indices = np.array([], dtype="int")

        for _ in range(5):
            try:
                indices = self._get_indices(segment_lengths, y, True)
            except ValueError:
                continue

            noise = self.dataset_generator.normal(0, 1 / signal_to_noise, (len(indices), X.shape[1]))

            changepoints = np.append([0], segment_lengths.cumsum())
            time_series = data.iloc[indices].drop(columns=class_label).to_numpy() + noise

            return  time_series,list(changepoints)

        raise ValueError("Not enough data")
        
    def generate_change_in_mean(self,n=600,d=5,mu=2):
        X = self.dataset_generator.normal(0, 1, (n, d))
        X[int(n / 3) : int(2 * n / 3), :] += mu
        orginal_change_points = [0, n / 3, 2 * n / 3, n]
        return X,orginal_change_points
    
    def generate_change_in_variance(self,n=600,d=5,std=[1,3,1]):
        segment_1 = self.dataset_generator.normal(0,std[0],(n//3,d))
        segment_2 = self.dataset_generator.normal(0,std[1],(n//3,d))
        segment_3 = self.dataset_generator.normal(0,std[0],(n//3,d))
        X = np.concatenate( (segment_1,segment_2,segment_3,),axis=0,)
        return X,[0,200,400,n]
    
    def generate_change_in_covariance(self,n=600,d=5,rho=0.7):
        Sigma = np.full((d, d), rho)
        np.fill_diagonal(Sigma, 1)
        segment_1 = self.dataset_generator.normal(0, 1, (n//3, d))
        segment_2 = self.dataset_generator.multivariate_normal(np.zeros(d),Sigma,n//3,method='cholesky')
        segment_3 = self.dataset_generator.normal(0, 1, (n//3, d))
        X = np.concatenate( (segment_1,segment_2,segment_3,),axis=0,)
        orginal_change_points = [0,200,400,n]
        return X,orginal_change_points
    
    def generate_dirichlet(self,d=20,n_segments=None,n_observations=None,minimal_relative_segment_length=None,
                           change_points=[0, 100, 130, 220, 320, 370, 520, 620, 740, 790, 870, 1000]):
        if n_segments is not None or n_observations is not None:
            if minimal_relative_segment_length is None:
                minimal_relative_segment_length = 1 / n_segments / 10
            segment_sizes = self._exponential_segment_lengths(n_segments, n_observations, minimal_relative_segment_length)
            changepoints = list(np.array([0] + segment_sizes.cumsum()))
        else:
            changepoints = change_points
        d = 20
        n_segments = len(changepoints) - 1
        params = self.dataset_generator.uniform(0, 0.2, n_segments * d).reshape((n_segments, d))

        X = np.zeros((changepoints[-1], d))
        for idx, (start, end) in enumerate(zip(changepoints[:-1], changepoints[1:])):
            X[start:end, :] = self.dataset_generator.dirichlet(params[idx, :], end - start)
        return X,changepoints
    
    def get_generated_data(self,name):
        if name == "CIM":
            return self.generate_change_in_mean()
        elif name == "CIV":
            return self.generate_change_in_variance()
        elif name == "CIC":
            return self.generate_change_in_covariance()
        elif name == "Dirichlet":
            return self.generate_dirichlet()
        else:
            raise ValueError("Incorrect Data")
        
    
    def get_all_data(self,file_paths):
        dataset_names = ['CIM',"CIV","CIC","Dirichlet"]
        data = []
        change_points = []
        for name in dataset_names:
            X,cpts = self.get_generated_data(name)
            data.append(X)
            change_points.append(cpts)
        for filepath in file_paths:
            X,cpts = self.get_data_from_csv(filepath)
            data.append(X)
            change_points.append(cpts)
        names_ = [i.split("/")[-1].split(".")[0] for i in file_paths]
        
        return data,change_points, dataset_names+names_
    def get_all_no_change_data(self,file_paths,class_labels=None):
        if class_labels == None:
            class_labels = ['class']*len(file_paths)
        dataset_names = ['CIM',"CIV","CIC","Dirichlet"]
        inds = [(0,200),(0,200),(0,200),(370,520)]
        data = []
        for name, ind in zip(dataset_names,inds):
            X,_ = self.get_generated_data(name)
            data.append(X[ind[0]:ind[1],:])
        for file_path,class_label in zip(file_paths,class_labels):
            df = pd.read_csv(file_path)
            values, counts = np.unique(df[class_label], return_counts=True)
            most_frequent = values[np.argmax(counts)]
            X = (df[lambda x: x[class_label] == most_frequent].drop(columns=class_label).to_numpy())
            self.dataset_generator.shuffle(X)
            data.append(X)
        names_ = [i.split("/")[-1].split(".")[0] for i in file_paths]
        
        return data, dataset_names+names_
        
    
            
                
                
        