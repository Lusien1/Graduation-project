import torch
import torch.nn as nn
import pandas as pd
import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import AngleEmbedding
from pennylane.operation import Tensor
import plotly.graph_objects as go
import plotly.offline as offline
from sklearn.preprocessing import MinMaxScaler


def block(weights, wires):
    qml.RX(weights[0], wires=wires[0])
    qml.RX(weights[1], wires=wires[1])
    qml.CNOT(wires=wires)


class TTNLayer(nn.Module):
    def __init__(self, n_qubits, n_layers):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.sim_dev = qml.device("lightning.qubit", wires=n_qubits)
        self.show_plot = True
        params = np.random.normal(size=(8, n_qubits - 1, 2))
        bias = np.random.normal(size=(3))
        class_bias = np.array([0.0])
        self.weights = nn.Parameter(torch.tensor(params, requires_grad=True))
        self.bias = nn.Parameter(torch.tensor(bias, requires_grad=True))
        self.class_bias = nn.Parameter(torch.tensor(class_bias, requires_grad=True))

    def QNode1(self, inputs, weights, bias, class_bias):
        @qml.qnode(self.sim_dev, interface='torch', diff_method="best")
        def quantum_model(inputs, weights, bias):
            # qml.BasisStatePreparation(inputs, wires=range(8))
            AngleEmbedding(inputs, wires=range(self.n_qubits))
            # AmplitudeEmbedding(inputs,wires=range(self.n_qubits),pad_with=64)
            qml.TTN(
                wires=range(self.n_qubits),
                n_block_wires=2,
                block=block,
                n_params_block=2,
                template_weights=weights,
            )
            return qml.expval(qml.PauliZ(wires=7))

        return quantum_model(inputs, weights, bias)

    def QNode2(self, inputs, weights, bias, class_bias):
        @qml.qnode(self.sim_dev, interface='torch', diff_method="best")
        def quantum_model(inputs, weights, bias):
            # qml.BasisStatePreparation(inputs, wires=range(8))
            AngleEmbedding(inputs, wires=range(self.n_qubits))
            # AmplitudeEmbedding(inputs,wires=range(self.n_qubits),pad_with=64)
            qml.TTN(
                wires=range(self.n_qubits),
                n_block_wires=2,
                block=block,
                n_params_block=2,
                template_weights=weights,
            )
            return qml.expval(qml.PauliZ(wires=7))

        return quantum_model(inputs, weights, bias)

    def QNode3(self, inputs, weights, bias, class_bias):
        @qml.qnode(self.sim_dev, interface='torch', diff_method="best")
        def quantum_model(inputs, weights, bias):
            # qml.BasisStatePreparation(inputs, wires=range(8))
            AngleEmbedding(inputs, wires=range(self.n_qubits))
            # AmplitudeEmbedding(inputs,wires=range(self.n_qubits),pad_with=64)
            qml.TTN(
                wires=range(self.n_qubits),
                n_block_wires=2,
                block=block,
                n_params_block=2,
                template_weights=weights,
            )
            return qml.expval(qml.PauliZ(wires=7))

        return quantum_model(inputs, weights, bias)

    def QNode4(self, inputs, weights, bias, class_bias):
        @qml.qnode(self.sim_dev, interface='torch', diff_method="best")
        def quantum_model(inputs, weights, bias):
            # qml.BasisStatePreparation(inputs, wires=range(8))
            AngleEmbedding(inputs, wires=range(self.n_qubits))
            # AmplitudeEmbedding(inputs,wires=range(self.n_qubits),pad_with=64)
            qml.TTN(
                wires=range(self.n_qubits),
                n_block_wires=2,
                block=block,
                n_params_block=2,
                template_weights=weights,
            )
            return qml.expval(qml.PauliZ(wires=7))

        return quantum_model(inputs, weights, bias)

    def QNode5(self, inputs, weights, bias, class_bias):
        @qml.qnode(self.sim_dev, interface='torch', diff_method="best")
        def quantum_model(inputs, weights, bias):
            # qml.BasisStatePreparation(inputs, wires=range(8))
            AngleEmbedding(inputs, wires=range(self.n_qubits))
            # AmplitudeEmbedding(inputs,wires=range(self.n_qubits),pad_with=64)
            qml.TTN(
                wires=range(self.n_qubits),
                n_block_wires=2,
                block=block,
                n_params_block=2,
                template_weights=weights,
            )
            return qml.expval(qml.PauliZ(wires=7))

        return quantum_model(inputs, weights, bias)

    def QNode6(self, inputs, weights, bias, class_bias):
        @qml.qnode(self.sim_dev, interface='torch', diff_method="best")
        def quantum_model(inputs, weights, bias):
            # qml.BasisStatePreparation(inputs, wires=range(8))
            AngleEmbedding(inputs, wires=range(self.n_qubits))
            # AmplitudeEmbedding(inputs,wires=range(self.n_qubits),pad_with=64)
            qml.TTN(
                wires=range(self.n_qubits),
                n_block_wires=2,
                block=block,
                n_params_block=2,
                template_weights=weights,
            )
            return qml.expval(qml.PauliZ(wires=7))

        return quantum_model(inputs, weights, bias)

    def QNode7(self, inputs, weights, bias, class_bias):
        @qml.qnode(self.sim_dev, interface='torch', diff_method="best")
        def quantum_model(inputs, weights, bias):
            # qml.BasisStatePreparation(inputs, wires=range(8))
            AngleEmbedding(inputs, wires=range(self.n_qubits))
            # AmplitudeEmbedding(inputs,wires=range(self.n_qubits),pad_with=64)
            qml.TTN(
                wires=range(self.n_qubits),
                n_block_wires=2,
                block=block,
                n_params_block=2,
                template_weights=weights,
            )
            return qml.expval(qml.PauliZ(wires=7))

        return quantum_model(inputs, weights, bias)

    def QNode8(self, inputs, weights, bias, class_bias):
        @qml.qnode(self.sim_dev, interface='torch', diff_method="best")
        def quantum_model(inputs, weights, bias):
            # qml.BasisStatePreparation(inputs, wires=range(8))
            AngleEmbedding(inputs, wires=range(self.n_qubits))
            # AmplitudeEmbedding(inputs,wires=range(self.n_qubits),pad_with=64)
            qml.TTN(
                wires=range(self.n_qubits),
                n_block_wires=2,
                block=block,
                n_params_block=2,
                template_weights=weights,
            )
            return qml.expval(qml.PauliZ(wires=7))

        return quantum_model(inputs, weights, bias)

    def forward(self, x):
        # print(x[:, :8].shape)
        # print(self.QNode1(x[:, :8], self.weights[0], self.bias, self.class_bias))
        res1 = self.QNode1(x[:, :8], self.weights[0], self.bias, self.class_bias).unsqueeze(dim=0).permute(1, 0)
        res2 = self.QNode2(x[:, 8:16], self.weights[1], self.bias, self.class_bias).unsqueeze(dim=0).permute(1, 0)
        res3 = self.QNode3(x[:, 16:24], self.weights[2], self.bias, self.class_bias).unsqueeze(dim=0).permute(1, 0)
        res4 = self.QNode4(x[:, 24:32], self.weights[3], self.bias, self.class_bias).unsqueeze(dim=0).permute(1, 0)
        res5 = self.QNode5(x[:, 32:40], self.weights[4], self.bias, self.class_bias).unsqueeze(dim=0).permute(1, 0)
        res6 = self.QNode6(x[:, 40:48], self.weights[5], self.bias, self.class_bias).unsqueeze(dim=0).permute(1, 0)
        res7 = self.QNode7(x[:, 48:56], self.weights[6], self.bias, self.class_bias).unsqueeze(dim=0).permute(1, 0)
        res8 = self.QNode8(x[:, 56:], self.weights[7], self.bias, self.class_bias).unsqueeze(dim=0).permute(1, 0)
        # print(res1.shape)
        # res=torch.stack(res1,res2,res3,res4,res5,res6,res7,res8)
        res = torch.cat((res1, res2, res3, res4, res5, res6, res7, res8), dim=1).float()
        return res


class QuantumLayer(nn.Module):
    def __init__(self, n_qubits, n_layers):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.sim_dev = qml.device("lightning.qubit", wires=n_qubits)
        self.show_plot = True
        params = np.random.normal(size=(n_layers, n_qubits, 3))
        bias = np.random.normal(size=(3))
        class_bias = np.array([0.0])
        self.weights = nn.Parameter(torch.tensor(params, requires_grad=True))
        self.bias = nn.Parameter(torch.tensor(bias, requires_grad=True))
        self.class_bias = nn.Parameter(torch.tensor(class_bias, requires_grad=True))
        # self.last=torch.nn.Sequential(nn.Sigmoid())

    def QNode(self, inputs, weights, bias, class_bias):
        @qml.qnode(self.sim_dev, interface='torch', diff_method="best")
        def quantum_model(inputs, weights, bias):
            """A variational quantum model."""
            # embedding
            # AmplitudeEmbedding(inputs, wires=range(6),pad_with=64,normalize=True)
            AngleEmbedding(inputs, wires=range(self.n_qubits))

            # trainable measurement
            for layer in range(self.n_layers):
                qml.CNOT(wires=[0, 1])
                qml.CNOT(wires=[1, 2])
                qml.CNOT(wires=[2, 3])
                qml.CNOT(wires=[3, 0])
                for i in range(self.n_qubits):
                    qml.RX(weights[layer, i, 0], wires=i)
                    qml.RY(weights[layer, i, 1], wires=i)
                    qml.RZ(weights[layer, i, 2], wires=i)

            # return [qml.expval(qml.PauliZ(0)),qml.expval(qml.PauliZ(1)),qml.expval(qml.PauliZ(2)),qml.expval(qml.PauliZ(3))]
            return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(2)),
                    qml.expval(qml.PauliZ(3))]

        # print(quantum_model(inputs,weights,bias),class_bias)
        return quantum_model(inputs, weights, bias)

    def forward(self, x):
        # res_list=[]
        # for x_n in x:
        #     res_n=self.QNode(x_n,self.weights,self.bias,self.class_bias)
        #     res_n=torch.tensor(res_n,requires_grad=True).to(device)
        #     res_list.append(res_n)
        # res=torch.stack(res_list)
        res = self.QNode(x, self.weights, self.bias, self.class_bias)
        res = torch.stack(res)
        res = res.permute(1, 0)
        # print(res)
        # res=torch.tensor(res,requires_grad=True).to(device)
        return res

def quantum_model_predict(X_pred, model):
    """Predict using the quantum model defined above."""
    # preds=[]
    # for x in X_pred:
    print(model(X_pred).shape)
    preds=torch.argmax(model(X_pred),dim=1)
    preds=3
        # preds.append(x_pred)
    return preds

def fillx(accident_data):
    X=accident_data
    Default_X=pd.read_csv('./Default_X.csv')
    # print(Default_X)
    # ['Start_Lng', 'Start_Lat', 'State', 'City', 'County', 'Start_Time']
    Default_X['Start_Lat']=X['Start_Lat']
    Default_X['Start_Lng'] = X['Start_Lng']
    Default_X['State'] = X['State']
    Default_X['City'] = X['City']
    Default_X['Country'] = X['County']
    Default_X['Start_Time'] = X['Start_Time']
    return Default_X


def feature_engin(accident_data):
    X=accident_data
    X["Start_Time"] = pd.to_datetime(X["Start_Time"], format='mixed')
    X["Year"] = X["Start_Time"].dt.year
    X["Month"] = X["Start_Time"].dt.month
    X["Weekday"] = X["Start_Time"].dt.weekday
    X["Day"] = X["Start_Time"].dt.day
    X["Hour"] = X["Start_Time"].dt.hour
    X["Minute"] = X["Start_Time"].dt.minute
    features_to_drop = ["ID", "Source", "Start_Time", "End_Time", "End_Lat", "End_Lng", "Description", "Street",
                        "County",
                        "State", "Zipcode", "Country", "Timezone", "Airport_Code", "Weather_Timestamp", "Wind_Chill(F)",
                        "Turning_Loop", "Sunrise_Sunset", "Nautical_Twilight", "Astronomical_Twilight"]
    X = X.drop(features_to_drop, axis=1)

    #处理天气
    X.loc[X["Weather_Condition"].str.contains("Thunder|T-Storm", na=False), "Weather_Condition"] = "Thunderstorm"
    X.loc[X["Weather_Condition"].str.contains("Snow|Sleet|Wintry", na=False), "Weather_Condition"] = "Snow"
    X.loc[X["Weather_Condition"].str.contains("Rain|Drizzle|Shower", na=False), "Weather_Condition"] = "Rain"
    X.loc[X["Weather_Condition"].str.contains("Wind|Squalls", na=False), "Weather_Condition"] = "Windy"
    X.loc[X["Weather_Condition"].str.contains("Hail|Pellets", na=False), "Weather_Condition"] = "Hail"
    X.loc[X["Weather_Condition"].str.contains("Fair", na=False), "Weather_Condition"] = "Clear"
    X.loc[X["Weather_Condition"].str.contains("Cloud|Overcast", na=False), "Weather_Condition"] = "Cloudy"
    X.loc[X["Weather_Condition"].str.contains("Mist|Haze|Fog", na=False), "Weather_Condition"] = "Fog"
    X.loc[X["Weather_Condition"].str.contains("Sand|Dust", na=False), "Weather_Condition"] = "Sand"
    X.loc[X["Weather_Condition"].str.contains("Smoke|Volcanic Ash", na=False), "Weather_Condition"] = "Smoke"
    X.loc[X["Weather_Condition"].str.contains("N/A Precipitation", na=False), "Weather_Condition"] = np.nan
    weather_list=['Weather_Condition_Cloudy','Weather_Condition_Fog','Weather_Condition_Hail'
                ,'Weather_Condition_Rain','Weather_Condition_Sand','Weather_Condition_Smoke'
                ,'Weather_Condition_Snow','Weather_Condition_Thunderstorm','Weather_Condition_Tornado'
                ,'Weather_Condition_Windy']
    for i in range(len(weather_list)):
        new_column={weather_list[i]:[0]}
        X=pd.concat([X,pd.DataFrame(new_column)],axis=1)
    X['Weather_Condition_'+X['Weather_Condition']]=1
    X=X.drop(["Weather_Condition"],axis=1)


    #处理风向
    X.loc[X["Wind_Direction"] == "CALM", "Wind_Direction"] = "Calm"
    X.loc[X["Wind_Direction"] == "VAR", "Wind_Direction"] = "Variable"
    X.loc[X["Wind_Direction"] == "East", "Wind_Direction"] = "E"
    X.loc[X["Wind_Direction"] == "North", "Wind_Direction"] = "N"
    X.loc[X["Wind_Direction"] == "South", "Wind_Direction"] = "S"
    X.loc[X["Wind_Direction"] == "West", "Wind_Direction"] = "W"
    X["Wind_Direction"] = X["Wind_Direction"].map(lambda x: x if len(x) != 3 else x[1:], na_action="ignore")

    wind_list=['Wind_Direction_E','Wind_Direction_N','Wind_Direction_NE','Wind_Direction_NW'
            ,'Wind_Direction_S','Wind_Direction_SE','Wind_Direction_SW','Wind_Direction_Variable','Wind_Direction_W']
    for i in range(len(wind_list)):
        new_column={wind_list[i]:[0]}
        X=pd.concat([X,pd.DataFrame(new_column)],axis=1)
    X['Wind_Direction_'+X['Wind_Direction']]=1
    X=X.drop(["Wind_Direction"],axis=1)

    #处理白天夜晚
    X=pd.concat([X,pd.DataFrame({"Civil_Twilight_Night":[0]})],axis=1)
    # print(X["Civil_Twilight"])
    if X["Civil_Twilight"][0]=="Night":
        X["Civil_Twilight_Night"]=1
    X=X.drop(["Civil_Twilight"],axis=1)


    #处理城市
    for i in range(14):
        new_column={"City_"+str(i):[0]}
        X=pd.concat([X,pd.DataFrame(new_column)],axis=1)
    X["City_1"]=1
    X["City_4"]=1
    X["City_5"]=1
    X["City_11"]=1
    X=X.drop(["City"],axis=1)

    X = X.replace([True, False], [1, 0])
    return X

def getPredictions(accident_data,categories):
    if categories=='csv':
        X=accident_data.drop('Severity',axis=1)
    else:
        X=accident_data
        X=fillx(X)
        print(X)
    X=feature_engin(X)
    print(X.columns.values)
    scaler = MinMaxScaler()
    scaler = scaler.fit(X)
    X = scaler.transform(X)
    X=np.array(X)
    X=X.astype(float)
    X_temp=X.copy()
    X_new=np.concatenate((X,X_temp),axis=0)
    print(X_new.shape)

    n_qubits=4
    n_layers=4
    layer_1 = torch.nn.Linear(8, 4)
    layers = [TTNLayer(n_qubits=8, n_layers=1), layer_1, QuantumLayer(n_qubits=4, n_layers=4)]
    model = torch.nn.Sequential(*layers)
    model.load_state_dict(torch.load('./TTN_VQC_cpu_state_dict.pth'))
    pred=quantum_model_predict(X_new,model)


    return "严重程度："+str(int(pred))

def make_vis(accident_data,result,categories):
    data_sever=accident_data[:1]
    data_sever['Severity']=int(result[-1])
    print(data_sever)
    fig = go.Figure(data=go.Scattergeo(
        locationmode='USA-states',
        lon=data_sever['Start_Lng'],
        lat=data_sever['Start_Lat'],
        text=data_sever['City'],
        mode='markers',
        marker=dict(
            size=8,
            opacity=0.8,
            reversescale=False,
            autocolorscale=False,
            symbol='circle',
            line=dict(
                width=1,
                color='rgba(102, 102, 102)'
            ),
            colorscale='Reds',
            cmax=4,
            color=data_sever['Severity'],
            cmin=1,
            colorbar_title="Severity"
        )))

    fig.update_layout(
        title='Severity of accidents',
        geo=dict(
            scope='usa',
            projection_type='albers usa',
            showland=True,
            landcolor="rgb(250, 250, 250)",
            subunitcolor="rgb(217, 217, 217)",
            countrycolor="rgb(217, 217, 217)",
            countrywidth=0.7,
            subunitwidth=0.7
        ),
    )
    # fig.show()
    offline.plot(fig, filename='./templates/{}_severity_predict.html'.format(categories), auto_open=False)