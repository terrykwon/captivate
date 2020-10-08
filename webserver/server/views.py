from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib.auth.models import User

from django.views.decorators.csrf import csrf_exempt 

# Create your views here.
def hello_world(request):
    pass


@csrf_exempt
def user_login(request):
    if request.method == 'GET':
        return render(request, 'server/login.html')
    elif request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']

        user = authenticate(request, 
                username=username, password=password)
        if user is not None:
            print('user successfully logged in!')
            login(request, user)

            return redirect('main', username=user.get_username())
        else:
            print('failed to log in!')
            return redirect('login')


def user_detail(request, username) :
    user = User.objects.get(username=username)

    context = {
        'user': user
    }

    return render(request, 'server/user-detail.html', context)


def main(request, username):
    user = User.objects.get(username=username)

    context = {
        'user': user
    }

    return render(request, 'server/main.html', context)