from django.shortcuts import render
from django.db.models import Q
from django.forms import model_to_dict
from . import models
from django.http import HttpResponse,HttpResponseRedirect
import json
from datetime import date
import datetime
from django.contrib import messages

name_list=['id','source','severity','start_time','end_time','start_lat','start_lng','end_lat','end_lng','distance_mi_field'
           ,'description','street','city','county','state','zipcode','country','timezone','airport_code','weather_timestamp'
           ,'temperature_f_field','wind_chill_f_field','pressure_in_field','visibility_mi_field'
           ,'wind_direction','wind_speed_mph_field','precipitation_in_field','weather_condition','amenity','bump'
           ,'crossing','give_way','junction','no_exit','railway','roundabout','station','stop','traffic_calming'
           ,'traffic_signal','turning_loop','sunrise_sunset','civil_twilight','nautical_twilight','astronomical_twilight']
# Create your views here.


def table_index(request):
    # info1=models.UsAccidentsMarch23.objects.values('id','source','severity','start_time','end_time',
    #                                                'start_lat','start_lng','end_lat','end_lng','distance_mi_field'
    #        ,'description','street','city','county','state','zipcode','country','timezone','airport_code','weather_timestamp'
    #         ,'temperature_f_field','wind_chill_f_field','pressure_in_field','visibility_mi_field'
    #        ,'wind_direction','wind_speed_mph_field','precipitation_in_field','weather_condition','amenity','bump'
    #        ,'crossing','give_way','junction','no_exit','railway','roundabout','station','stop','traffic_calming'
    #        ,'traffic_signal','turning_loop','sunrise_sunset','civil_twilight','nautical_twilight','astronomical_twilight')
    #
    # info1=info1[:100]
    filter_field=['id','source','severity','state','city','county','zipcode']
    filter_dict={}
    clear_button=request.POST.get('clear_button')
    # print(clear_button)
    for field in filter_field:
        value=request.POST.get(field)
        # print(value)
        if value:
            filter_dict[field]=value

    if clear_button!=None:
        filter_dict={}
    search=request.POST.get('search')

    date_dict={}
    value_start=request.POST.get('start_date')
    value_end=request.POST.get('end_date')
    date_dict['value_start']=value_start
    date_dict['value_end']=value_end
    # print(value_start, value_end)
    if date_dict or clear_button!=None:
        # print(value_start, value_end)
        fmt = '%Y-%m-%d'
        if date_dict!=None and (date_dict['value_start']==None or date_dict['value_start']==''):
            with open("./date_json.json", encoding="utf-8") as f:
                date_dict = json.load(f)
        if clear_button!=None:
            date_dict['value_start']='2016-01-01'
            date_dict['value_end']='2023-03-31'
        print(date_dict)
        info_search_time=models.UsAccidentsMarch23.objects.filter(start_time__range = (datetime.datetime.strptime(date_dict['value_start'], fmt),datetime.datetime.strptime(date_dict['value_end'], fmt))).values('id','source','severity','start_time','end_time',
                                                   'start_lat','start_lng','end_lat','end_lng','distance_mi_field'
           ,'description','street','city','county','state','zipcode','country','timezone','airport_code','weather_timestamp'
            ,'temperature_f_field','wind_chill_f_field','pressure_in_field','visibility_mi_field'
           ,'wind_direction','wind_speed_mph_field','precipitation_in_field','weather_condition','amenity','bump'
           ,'crossing','give_way','junction','no_exit','railway','roundabout','station','stop','traffic_calming'
           ,'traffic_signal','turning_loop','sunrise_sunset','civil_twilight','nautical_twilight','astronomical_twilight')
        with open("./date_json.json", "w", encoding='utf-8') as f:
            json.dump(date_dict, f, indent=2, sort_keys=True, ensure_ascii=False)  # 写为多行
    else:
        with open("./date_json.json", encoding="utf-8") as f:
            date_dict = json.load(f)
        info_search_time = models.UsAccidentsMarch23.objects.filter(start_time__range = (datetime.datetime.strptime(date_dict['value_start'], fmt),datetime.datetime.strptime(date_dict['value_end'], fmt))).values('id', 'source', 'severity',
                                                                                     'start_time', 'end_time',
                                                                                     'start_lat', 'start_lng',
                                                                                     'end_lat', 'end_lng',
                                                                                     'distance_mi_field'
                                                                                     , 'description', 'street', 'city',
                                                                                     'county', 'state', 'zipcode',
                                                                                     'country', 'timezone',
                                                                                     'airport_code', 'weather_timestamp'
                                                                                     , 'temperature_f_field',
                                                                                     'wind_chill_f_field',
                                                                                     'pressure_in_field',
                                                                                     'visibility_mi_field'
                                                                                     , 'wind_direction',
                                                                                     'wind_speed_mph_field',
                                                                                     'precipitation_in_field',
                                                                                     'weather_condition', 'amenity',
                                                                                     'bump'
                                                                                     , 'crossing', 'give_way',
                                                                                     'junction', 'no_exit', 'railway',
                                                                                     'roundabout', 'station', 'stop',
                                                                                     'traffic_calming'
                                                                                     , 'traffic_signal', 'turning_loop',
                                                                                     'sunrise_sunset', 'civil_twilight',
                                                                                     'nautical_twilight',
                                                                                     'astronomical_twilight')


    if filter_dict or clear_button!=None:
        # print(value_start, value_end)
        fmt = '%Y-%m-%d'
        # start_time__range = (datetime.datetime.strptime(date_dict['value_start'], fmt),datetime.datetime.strptime(date_dict['value_end'], fmt))
        info_search=models.UsAccidentsMarch23.objects.filter(**filter_dict).values('id','source','severity','start_time','end_time',
                                                   'start_lat','start_lng','end_lat','end_lng','distance_mi_field'
           ,'description','street','city','county','state','zipcode','country','timezone','airport_code','weather_timestamp'
            ,'temperature_f_field','wind_chill_f_field','pressure_in_field','visibility_mi_field'
           ,'wind_direction','wind_speed_mph_field','precipitation_in_field','weather_condition','amenity','bump'
           ,'crossing','give_way','junction','no_exit','railway','roundabout','station','stop','traffic_calming'
           ,'traffic_signal','turning_loop','sunrise_sunset','civil_twilight','nautical_twilight','astronomical_twilight')
        with open("./write_json.json", "w", encoding='utf-8') as f:
            json.dump(filter_dict, f, indent=2, sort_keys=True, ensure_ascii=False)  # 写为多行
        # if date_dict or clear_button!=None:
        #     with open("./date_json.json", "w", encoding='utf-8') as f:
        #         json.dump(date_dict, f, indent=2, sort_keys=True, ensure_ascii=False)  # 写为多行
    elif search:
        info_search=models.UsAccidentsMarch23.objects.filter(
            Q(id__contains=search) | Q(source__contains=search) |
            Q(city__contains=search) | Q(county__contains=search) |
            Q(state__contains=search) | Q(zipcode__contains=search)
        ).values('id','source','severity','start_time','end_time',
                                                   'start_lat','start_lng','end_lat','end_lng','distance_mi_field'
           ,'description','street','city','county','state','zipcode','country','timezone','airport_code','weather_timestamp'
            ,'temperature_f_field','wind_chill_f_field','pressure_in_field','visibility_mi_field'
           ,'wind_direction','wind_speed_mph_field','precipitation_in_field','weather_condition','amenity','bump'
           ,'crossing','give_way','junction','no_exit','railway','roundabout','station','stop','traffic_calming'
           ,'traffic_signal','turning_loop','sunrise_sunset','civil_twilight','nautical_twilight','astronomical_twilight')
    else:
        with open("./write_json.json", encoding="utf-8") as f:
            filter_dict = json.load(f)
        info_search=models.UsAccidentsMarch23.objects.filter(**filter_dict).values('id','source','severity','start_time','end_time',
                                                   'start_lat','start_lng','end_lat','end_lng','distance_mi_field'
           ,'description','street','city','county','state','zipcode','country','timezone','airport_code','weather_timestamp'
            ,'temperature_f_field','wind_chill_f_field','pressure_in_field','visibility_mi_field'
           ,'wind_direction','wind_speed_mph_field','precipitation_in_field','weather_condition','amenity','bump'
           ,'crossing','give_way','junction','no_exit','railway','roundabout','station','stop','traffic_calming'
           ,'traffic_signal','turning_loop','sunrise_sunset','civil_twilight','nautical_twilight','astronomical_twilight')


    info2=(info_search & info_search_time)
    # print(str(info2.query))
    info2_list=[]
    if info2:
        for row in info2:
            temp_dict={}
            for name in name_list:
                temp_dict.update({name:row[name]})
            info2_list.append(temp_dict)

    # try:
    page=int(request.GET.get('page',1))
    # print(page,type(page))
    if page<1:
        page=1
    # except Exception:
    #     page=1
    # print(page)

    page_num = 10 #每页数据条数
    total_num,remainder=divmod(len(info2_list),page_num)
    if remainder!=0:
        total_num+=1

    max_page_num = 7 #每页页码数

    #处理边界条件
    half_num = max_page_num // 2
    if total_num < max_page_num:
        page_start = 1
        page_end = total_num
    else:
        if page - half_num < 1:
            page_start = 1
            page_end = max_page_num
        elif page + half_num > total_num:
            page_start = total_num - max_page_num + 1
            page_end = total_num
        else:
            page_start = page - half_num
            page_end = page + half_num

    html_list = []
    html_list.append(
        '<li><a href="?page=%s"><span aria-hidden="true">首页</span></a></li>' % (1))

    if page == 1:
        html_list.append(
            '<li class="disabled"><a href="#" aria-label="Previous"><span aria-hidden="true">&laquo;</span></a></li>')
    else:
        html_list.append(
            '<li><a href="?page=%s" aria-label="Previous"><span aria-hidden="true">&laquo;</span></a></li>' % (
                        page - 1))

    for i in range(page_start, page_end + 1):
        if page == i:
            html_list.append('<li class="active"><a href="?page=%s">%s</a></li>' % (i, i))
        else:
            html_list.append('<li><a href="?page=%s">%s</a></li>' % (i, i))

    if page == total_num:
        html_list.append(
            '<li class="disabled"><a href="#" aria-label="Next"><span aria-hidden="true">&raquo;</span></a></li>')
    else:
        html_list.append(
            '<li><a href="?page=%s" aria-label="Next"><span aria-hidden="true">&raquo;</span></a></li>'
            % (page + 1))

    html_list.append(
        '<li><a href="?page=%s"><span aria-hidden="true">尾页</span></a></li>' % (total_num))

    html_list = ''.join(html_list)

    start=(page-1)*page_num
    end=page*page_num

    context = {'info2': info2_list[start:end], 'info_name': name_list,'html_list':html_list,'fields':filter_dict}

    return render(request, 'table.html', context)


city_list=['Miami','Houston','Los Angeles','Charlotte','Dallas','Orlando','Austin','Raleigh']


def register(request):
    if request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        user = models.User()
        user.username = username
        user.email = email
        user.password = password
        user.save()
    return render(request,'register.html')


def login(request):
    if request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        user = models.User.objects.filter(email=email)
        if user.exists():
        # print(password,user.password)
            user0=models.User.objects.get(email=email)
            if username == user0.username:
                if password == user0.password:
                    return render(request,'index.html')
                else:
                    messages.error(request, '密码错误！')
                    return render(request,'login.html',{"error":"密码错误"})
            else:
                return render(request,'login.html',{"usernameerror":"用户名不存在"})
        else:
            return render(request,'login.html',{"emailerror":"邮箱未注册，请先去注册"})
    return render(request,'login.html')
