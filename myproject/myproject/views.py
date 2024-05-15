from django.shortcuts import render
from django.http import HttpResponse
from myproject import predict_severity as ps
import pandas as pd
import json

# Create your views here.
# our home page view
def home(request):
    #初始界面
    return render(request, 'index.html')

def sample_10000(request):
    return render(request,'sample_10000.html')

def sample_10000_visibility_severity(request):
    return render(request,'sample_10000_visibility_severity.html')

def sample_10000_distance_severity(request):
    return render(request,'sample_10000_distance_severity.html')

def user_information(request):
    with open("./temp_information.json", encoding="utf-8") as f:
        filter_dict = json.load(f)
    filter_field=['username','email','fname','lname','address','city','country','zipcode','description']
    # print(clear_button)
    for field in filter_field:
        value=request.POST.get(field)
        # print(value)
        if value:
            filter_dict[field]=value
    with open("./temp_information.json", "w", encoding='utf-8') as f:
        json.dump(filter_dict, f, indent=2, sort_keys=True, ensure_ascii=False)

    if request.method == 'POST':
        file = request.FILES.get('photo_file')
        # print("00000000000")
        if file!=None:
            content=file.chunks()
            with open('./static/img/faces/myphoto.jpg', 'wb') as f:
                for i in content:
                    f.write(i)
    return render(request,'user_information.html',{'information':filter_dict})

# our result page view
def severity_predict(request):
    return render(request,'severity_predict.html')

def result(request):
    #严重程度预测结果
    # distance=int(request.GET['distance'])
    # result=ps.getPredictions(distance)
    if request.method == 'POST':
        # try:
        f = request.FILES.get('csv_file')
        print(f.name)
        excel_type = f.name.split('.')[1]
        print(excel_type)
        if excel_type in ['csv']:
            accident_data = pd.read_csv(f)
            print("1111111")
            result=ps.getPredictions(accident_data,categories='csv')
            ps.make_vis(accident_data,result,categories='csv')
            return render(request, 'severity_predict.html', {'result': result})
    #     except:
    #         error = '解析excel文件或者数据插入错误'
    #         return render(request, 'severity_predict.html', {"error": error})
    # return render(request, 'severity_predict.html', {'result': result})
def csv_severity_predict(request):
    return render(request,'csv_severity_predict.html')

def result2(request):
    # print(request.method) #POST
    if request.method == 'POST':
        print(request.POST.get('State'))
        filter_field=['Start_Lng','Start_Lat','State','City','County','Start_Time']
        filter_dict={}
        for field in filter_field:
            value=request.POST.get(field)
            print(value)
            if value:
                filter_dict[field]=value
        print(filter_dict)
        accident_data=pd.DataFrame([filter_dict])
        result = ps.getPredictions(accident_data,categories='fields')
        ps.make_vis(accident_data, result,categories='fields')
        context={'result2':result,'fields':filter_dict}
        return render(request, 'severity_predict.html', context)


def fields_severity_predict(request):
    return render(request,'fields_severity_predict.html')


def sum_state(request):
    #事故总数可视化
    return render(request,'accident_sum_state.html')

#聚类
def clustering_location(request):
    #整体页面
    return render(request,'clustering_location.html')
def state_kmeans(request):
    #按州事故数聚类
    return render(request,'state_severity_kmeans.html')

def sample_1000(request):
    #采样1000个事故可视化
    return render(request,'sample_1000_.html')

def sample_1000_kmeans(request):
    return render(request,'sample_1000_kmeans.html')

def sample_1000_hierarchial(request):
    return render(request,'sample_1000_hierarchial.html')

def clustering_time(request):
    #整体页面
    return render(request,'clustering_time.html')

#时序
#展示图片
def time_series_city(request):
    #整体页面
    return render(request,'time_series_city.html')
def important_city_location(request):
    #重要城市位置
    return render(request,'important_city_location.html')
def city_model_change(request):
    #切换城市和模型时序预测
    city_name=str(request.GET['city_name'])
    model_name=str(request.GET['model_name'])
    imgstr='model_result/'+model_name+'/'+model_name+'_'+city_name+'.png'
    context={'imgstr':imgstr,'city_name':city_name,'model_name':model_name}
    # context = {'city_name': city_name, 'model_name': model_name}

    return render(request,'time_series_city.html',context)
